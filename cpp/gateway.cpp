// Native session-context gateway for MNEMOSYNE-MNS.
//
// This process intentionally owns only the latency-sensitive local data plane:
// authenticated HTTP ingress, SQLite persistence, and chronological session
// retrieval. Python keeps model training, ONNX export, experimental-global
// policy, and the provider fallback. See docs/NATIVE-GATEWAY.md.

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#ifndef MNEMOSYNE_GATEWAY_VERSION
#define MNEMOSYNE_GATEWAY_VERSION "1.0.0-native"
#endif

namespace {

constexpr int kSqliteOk = 0;
constexpr int kSqliteRow = 100;
constexpr int kSqliteDone = 101;
constexpr int kSqliteOpenReadWrite = 0x00000002;
constexpr int kSqliteOpenCreate = 0x00000004;
constexpr int kSqliteOpenFullMutex = 0x00010000;
constexpr int kSqliteTransient = -1;
constexpr std::size_t kMaxHeaderBytes = 32 * 1024;
constexpr std::size_t kMaxBodyBytes = 1024 * 1024;

struct sqlite3;
struct sqlite3_stmt;

class SqliteApi {
public:
    using open_v2_fn = int (*)(const char*, sqlite3**, int, const char*);
    using close_v2_fn = int (*)(sqlite3*);
    using exec_fn = int (*)(sqlite3*, const char*, int (*)(void*, int, char**, char**), void*, char**);
    using free_fn = void (*)(void*);
    using prepare_v2_fn = int (*)(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
    using bind_text_fn = int (*)(sqlite3_stmt*, int, const char*, int, void (*)(void*));
    using bind_double_fn = int (*)(sqlite3_stmt*, int, double);
    using step_fn = int (*)(sqlite3_stmt*);
    using finalize_fn = int (*)(sqlite3_stmt*);
    using column_text_fn = const unsigned char* (*)(sqlite3_stmt*, int);
    using column_double_fn = double (*)(sqlite3_stmt*, int);
    using changes_fn = int (*)(sqlite3*);
    using errmsg_fn = const char* (*)(sqlite3*);

    SqliteApi() {
        const char* configured = std::getenv("MNEMOSYNE_SQLITE_DLL");
        std::vector<std::string> candidates;
        if (configured != nullptr && *configured != '\0') {
            candidates.emplace_back(configured);
        }
        candidates.emplace_back("sqlite3.dll");
        for (const auto& candidate : candidates) {
            library_ = LoadLibraryA(candidate.c_str());
            if (library_ != nullptr) {
                break;
            }
        }
        if (library_ == nullptr) {
            throw std::runtime_error("SQLite runtime unavailable; set MNEMOSYNE_SQLITE_DLL to sqlite3.dll");
        }
        open_v2 = load<open_v2_fn>("sqlite3_open_v2");
        close_v2 = load<close_v2_fn>("sqlite3_close_v2");
        exec = load<exec_fn>("sqlite3_exec");
        free = load<free_fn>("sqlite3_free");
        prepare_v2 = load<prepare_v2_fn>("sqlite3_prepare_v2");
        bind_text = load<bind_text_fn>("sqlite3_bind_text");
        bind_double = load<bind_double_fn>("sqlite3_bind_double");
        step = load<step_fn>("sqlite3_step");
        finalize = load<finalize_fn>("sqlite3_finalize");
        column_text = load<column_text_fn>("sqlite3_column_text");
        column_double = load<column_double_fn>("sqlite3_column_double");
        changes = load<changes_fn>("sqlite3_changes");
        errmsg = load<errmsg_fn>("sqlite3_errmsg");
    }

    ~SqliteApi() {
        if (library_ != nullptr) {
            FreeLibrary(library_);
        }
    }

    SqliteApi(const SqliteApi&) = delete;
    SqliteApi& operator=(const SqliteApi&) = delete;

    open_v2_fn open_v2{};
    close_v2_fn close_v2{};
    exec_fn exec{};
    free_fn free{};
    prepare_v2_fn prepare_v2{};
    bind_text_fn bind_text{};
    bind_double_fn bind_double{};
    step_fn step{};
    finalize_fn finalize{};
    column_text_fn column_text{};
    column_double_fn column_double{};
    changes_fn changes{};
    errmsg_fn errmsg{};

private:
    template <typename T>
    T load(const char* symbol) {
        const auto proc = GetProcAddress(library_, symbol);
        if (proc == nullptr) {
            throw std::runtime_error(std::string("SQLite runtime is missing symbol: ") + symbol);
        }
        return reinterpret_cast<T>(proc);
    }

    HMODULE library_{};
};

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char c : value) {
        switch (c) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (c < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    output << "\\u00" << hex[(c >> 4) & 0x0f] << hex[c & 0x0f];
                } else {
                    output << static_cast<char>(c);
                }
        }
    }
    return output.str();
}

std::optional<std::string> json_string(const std::string& json, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    std::size_t position = json.find(marker);
    if (position == std::string::npos) {
        return std::nullopt;
    }
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) {
        return std::nullopt;
    }
    ++position;
    while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position]))) {
        ++position;
    }
    if (position >= json.size() || json[position] != '"') {
        return std::nullopt;
    }
    ++position;
    std::string result;
    while (position < json.size()) {
        const char c = json[position++];
        if (c == '"') {
            return result;
        }
        if (c != '\\') {
            result.push_back(c);
            continue;
        }
        if (position >= json.size()) {
            return std::nullopt;
        }
        const char escape = json[position++];
        switch (escape) {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u':
                // Protocol clients send UTF-8 directly. Rejecting escaped Unicode is
                // safer than silently mangling a session transcript.
                return std::nullopt;
            default: return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<int> json_int(const std::string& json, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    std::size_t position = json.find(marker);
    if (position == std::string::npos) return std::nullopt;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return std::nullopt;
    ++position;
    while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position]))) ++position;
    const std::size_t start = position;
    while (position < json.size() && std::isdigit(static_cast<unsigned char>(json[position]))) ++position;
    if (start == position) return std::nullopt;
    try { return std::stoi(json.substr(start, position - start)); } catch (...) { return std::nullopt; }
}

bool has_forbidden_visible_content(const std::string& text) {
    const std::string normalized = lowercase(text);
    static const std::vector<std::string> forbidden = {
        "<system", "</system", "<developer", "</developer", "# system prompt", "# system:",
        "<skill", "<skills", "skill_view(", "skill_manage(", "<thinking", "<reflection",
        "<antml:function_calls", "<memory-context", "[system note:", "[mnemosyne context]",
        "[previous conversation context]", "```thinking", "you are an ai", "you are an assistant",
        "you are an agent"
    };
    const std::string normalized_trimmed = lowercase(trim(text));
    if (trim(text).empty()) return true;
    for (const auto& marker : forbidden) {
        if (normalized.find(marker) != std::string::npos) return true;
    }
    return normalized_trimmed.rfind("system:", 0) == 0 || normalized_trimmed.rfind("developer:", 0) == 0;
}

double unix_time() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string stable_digest(const std::string& value) {
    // FNV-1a is stable across builds, unlike std::hash<std::string>.
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream output;
    output << std::hex << hash;
    return output.str();
}

int estimated_tokens(const std::string& utf8) {
    int count = 0;
    bool in_ascii_word = false;
    for (std::size_t i = 0; i < utf8.size();) {
        const unsigned char c = static_cast<unsigned char>(utf8[i]);
        if (c < 0x80) {
            if (std::isalnum(c) || c == '_') {
                if (!in_ascii_word) ++count;
                in_ascii_word = true;
            } else {
                in_ascii_word = false;
                if (!std::isspace(c)) ++count;
            }
            ++i;
            continue;
        }
        in_ascii_word = false;
        ++count;
        if ((c & 0xf0) == 0xf0) i += 4;
        else if ((c & 0xe0) == 0xe0) i += 3;
        else if ((c & 0xc0) == 0xc0) i += 2;
        else ++i;
    }
    return std::max(1, count);
}

class Statement {
public:
    Statement(SqliteApi& api, sqlite3* database, const char* sql) : api_(api) {
        if (api_.prepare_v2(database, sql, -1, &statement_, nullptr) != kSqliteOk) {
            throw std::runtime_error("SQLite statement preparation failed");
        }
    }
    ~Statement() { if (statement_ != nullptr) api_.finalize(statement_); }
    sqlite3_stmt* get() const { return statement_; }
private:
    SqliteApi& api_;
    sqlite3_stmt* statement_{};
};

class SessionStore {
public:
    SessionStore(SqliteApi& api, const std::string& path) : api_(api) {
        if (api_.open_v2(path.c_str(), &database_, kSqliteOpenReadWrite | kSqliteOpenCreate | kSqliteOpenFullMutex, nullptr) != kSqliteOk) {
            throw std::runtime_error("could not open SQLite database: " + path);
        }
        exec("PRAGMA foreign_keys = ON;");
        exec("PRAGMA journal_mode = WAL;");
        exec("PRAGMA synchronous = NORMAL;");
        exec("CREATE TABLE IF NOT EXISTS memories (memory_id TEXT PRIMARY KEY, content TEXT NOT NULL, payload_json TEXT NOT NULL, updated_at REAL NOT NULL, version INTEGER NOT NULL);");
        exec("CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(memory_id UNINDEXED, content, tokenize='unicode61');");
    }

    ~SessionStore() { if (database_ != nullptr) api_.close_v2(database_); }

    SessionStore(const SessionStore&) = delete;
    SessionStore& operator=(const SessionStore&) = delete;

    std::string record_turn(const std::string& session_id, const std::string& turn_id,
                            const std::string& project_id, const std::string& user_text,
                            const std::string& assistant_text) {
        if (session_id.empty() || turn_id.empty()) throw std::invalid_argument("session_id and turn_id are required");
        if (has_forbidden_visible_content(user_text) || has_forbidden_visible_content(assistant_text)) {
            throw std::invalid_argument("turn payload contains system/developer/skill/agent content");
        }
        return insert_context("session:" + session_id + ":" + turn_id, session_id, project_id,
                              "[Session context]\nUser: " + trim(user_text) + "\nAssistant: " + trim(assistant_text));
    }

    std::string record_checkpoint(const std::string& session_id, const std::string& turn_id,
                                  const std::string& project_id, const std::string& transcript) {
        if (session_id.empty() || turn_id.empty() || trim(transcript).empty()) {
            throw std::invalid_argument("session_id, turn_id and transcript are required");
        }
        const std::string memory_id = "session:" + session_id + ":checkpoint:" + turn_id + ":" + stable_digest(transcript);
        if (exists(memory_id)) return memory_id;
        supersede_checkpoints(session_id, memory_id);
        return insert_context(memory_id, session_id, project_id, "[Session checkpoint]\n" + trim(transcript));
    }

    struct ContextItem { std::string id; std::string content; double created_at; int token_cost; };

    std::vector<ContextItem> recent(const std::string& session_id, const std::string& project_id,
                                    const int max_tokens) {
        if (session_id.empty()) throw std::invalid_argument("session_id is required");
        (void)project_id;  // Session identity is the isolation boundary in session mode.
        std::lock_guard<std::mutex> lock(mutex_);
        Statement statement(api_, database_,
            "SELECT memory_id, content, updated_at FROM memories "
            "WHERE memory_id LIKE ? AND memory_id NOT LIKE ? "
            "ORDER BY updated_at ASC, memory_id ASC LIMIT 100;");
        const std::string prefix = "session:" + session_id + ":%";
        const std::string checkpoints = "session:" + session_id + ":checkpoint:%";
        bind(statement.get(), 1, prefix);
        bind(statement.get(), 2, checkpoints);
        std::vector<ContextItem> items;
        int used_tokens = 0;
        while (api_.step(statement.get()) == kSqliteRow) {
            const char* id = reinterpret_cast<const char*>(api_.column_text(statement.get(), 0));
            const char* content = reinterpret_cast<const char*>(api_.column_text(statement.get(), 1));
            if (id == nullptr || content == nullptr) continue;
            const std::string content_text(content);
            const int cost = estimated_tokens(content_text);
            if (used_tokens + cost > max_tokens) break;
            used_tokens += cost;
            items.push_back({id, content_text, api_.column_double(statement.get(), 2), cost});
        }
        return items;
    }

private:
    bool exists(const std::string& memory_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        Statement statement(api_, database_, "SELECT 1 FROM memories WHERE memory_id = ? LIMIT 1;");
        bind(statement.get(), 1, memory_id);
        return api_.step(statement.get()) == kSqliteRow;
    }

    void supersede_checkpoints(const std::string& session_id, const std::string& replacement_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        Statement select(api_, database_, "SELECT memory_id, payload_json, version FROM memories WHERE memory_id LIKE ?;");
        bind(select.get(), 1, "session:" + session_id + ":checkpoint:%");
        std::vector<std::tuple<std::string, std::string, int>> checkpoints;
        while (api_.step(select.get()) == kSqliteRow) {
            const char* id = reinterpret_cast<const char*>(api_.column_text(select.get(), 0));
            const char* payload = reinterpret_cast<const char*>(api_.column_text(select.get(), 1));
            if (id == nullptr || payload == nullptr) continue;
            checkpoints.emplace_back(id, payload, static_cast<int>(api_.column_double(select.get(), 2)));
        }
        const double now = unix_time();
        for (const auto& checkpoint : checkpoints) {
            const auto state = std::get<1>(checkpoint).find("\"state\":\"active\"");
            if (state == std::string::npos) continue;
            std::string updated_payload = std::get<1>(checkpoint);
            updated_payload.replace(state, std::strlen("\"state\":\"active\""), "\"state\":\"superseded\"");
            const auto superseded_by = updated_payload.find("\"superseded_by\":null");
            if (superseded_by != std::string::npos) {
                updated_payload.replace(superseded_by, std::strlen("\"superseded_by\":null"), "\"superseded_by\":\"" + json_escape(replacement_id) + "\"");
            }
            Statement update(api_, database_, "UPDATE memories SET payload_json = ?, updated_at = ?, version = ? WHERE memory_id = ?;");
            bind(update.get(), 1, updated_payload);
            if (api_.bind_double(update.get(), 2, now) != kSqliteOk || api_.bind_double(update.get(), 3, static_cast<double>(std::get<2>(checkpoint) + 1)) != kSqliteOk) {
                throw std::runtime_error("could not bind superseded checkpoint");
            }
            bind(update.get(), 4, std::get<0>(checkpoint));
            if (api_.step(update.get()) != kSqliteDone) throw std::runtime_error("could not supersede previous checkpoint");
        }
    }

    void exec(const char* sql) {
        char* error = nullptr;
        const int rc = api_.exec(database_, sql, nullptr, nullptr, &error);
        if (rc != kSqliteOk) {
            std::string message = error == nullptr ? "SQLite error" : error;
            if (error != nullptr) api_.free(error);
            throw std::runtime_error(message);
        }
    }

    void bind(sqlite3_stmt* statement, const int index, const std::string& value) {
        if (api_.bind_text(statement, index, value.c_str(), static_cast<int>(value.size()), reinterpret_cast<void (*)(void*)>(kSqliteTransient)) != kSqliteOk) {
            throw std::runtime_error("SQLite text binding failed");
        }
    }

    std::string insert_context(const std::string& id, const std::string& session_id,
                               const std::string& project_id, const std::string& content) {
        const double now = unix_time();
        const std::string payload = "{\"memory_id\":\"" + json_escape(id) + "\",\"content\":\"" + json_escape(content) +
            "\",\"created_at\":" + std::to_string(now) + ",\"updated_at\":" + std::to_string(now) +
            ",\"version\":1,\"scope\":{\"level\":\"session\",\"session_id\":\"" + json_escape(session_id) +
            "\",\"project_id\":\"" + json_escape(project_id) + "\",\"visibility\":[\"session\"]},"
            "\"authority\":{\"class\":\"inferred\",\"priority\":30,\"source\":\"imported\",\"reliability\":0.5},"
            "\"validity\":{\"state\":\"active\",\"last_verified_at\":" + std::to_string(now) +
            ",\"stale_after\":null,\"superseded_by\":null,\"conflict_with\":[]},"
            "\"retrieval\":{\"path\":\"api/input\",\"tags\":[\"api\"],\"embedding_model\":\"intfloat/multilingual-e5-small\",\"embedding_version\":1},"
            "\"usage\":{\"retrieved_count\":0,\"cited_count\":0,\"corrected_count\":0,\"last_retrieved_at\":null,\"last_cited_at\":null},"
            "\"egress\":{\"allowed_destinations\":[\"local\"],\"sensitivity\":\"private\",\"redaction_required\":false},"
            "\"provenance\":{\"created_by\":\"import\",\"import_source\":\"api\",\"parent_memory_id\":null}}";
        std::lock_guard<std::mutex> lock(mutex_);
        exec("BEGIN IMMEDIATE;");
        try {
            Statement insert(api_, database_, "INSERT OR IGNORE INTO memories(memory_id, content, payload_json, updated_at, version) VALUES (?, ?, ?, ?, 1);");
            bind(insert.get(), 1, id); bind(insert.get(), 2, content); bind(insert.get(), 3, payload);
            if (api_.bind_double(insert.get(), 4, now) != kSqliteOk || api_.step(insert.get()) != kSqliteDone) {
                throw std::runtime_error("could not write context memory");
            }
            if (api_.changes(database_) == 1) {
                Statement fts(api_, database_, "INSERT INTO memories_fts(memory_id, content) VALUES (?, ?);");
                bind(fts.get(), 1, id); bind(fts.get(), 2, content);
                if (api_.step(fts.get()) != kSqliteDone) throw std::runtime_error("could not update FTS index");
            }
            exec("COMMIT;");
        } catch (...) {
            try { exec("ROLLBACK;"); } catch (...) {}
            throw;
        }
        return id;
    }

    SqliteApi& api_;
    sqlite3* database_{};
    std::mutex mutex_;
};

struct Request { std::string method; std::string path; std::map<std::string, std::string> headers; std::string body; };

std::optional<Request> read_request(const SOCKET client) {
    std::string buffer;
    char chunk[4096];
    while (buffer.find("\r\n\r\n") == std::string::npos) {
        const int received = recv(client, chunk, static_cast<int>(sizeof(chunk)), 0);
        if (received <= 0) return std::nullopt;
        buffer.append(chunk, static_cast<std::size_t>(received));
        if (buffer.size() > kMaxHeaderBytes) return std::nullopt;
    }
    const std::size_t header_end = buffer.find("\r\n\r\n");
    std::istringstream lines(buffer.substr(0, header_end));
    std::string first_line;
    if (!std::getline(lines, first_line)) return std::nullopt;
    std::istringstream request_line(trim(first_line));
    Request request;
    std::string version;
    if (!(request_line >> request.method >> request.path >> version)) return std::nullopt;
    std::string line;
    while (std::getline(lines, line)) {
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        request.headers[lowercase(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
    }
    std::size_t content_length = 0;
    if (const auto it = request.headers.find("content-length"); it != request.headers.end()) {
        try { content_length = static_cast<std::size_t>(std::stoul(it->second)); } catch (...) { return std::nullopt; }
    }
    if (content_length > kMaxBodyBytes) return std::nullopt;
    request.body = buffer.substr(header_end + 4);
    while (request.body.size() < content_length) {
        const int received = recv(client, chunk, static_cast<int>(std::min(sizeof(chunk), content_length - request.body.size())), 0);
        if (received <= 0) return std::nullopt;
        request.body.append(chunk, static_cast<std::size_t>(received));
    }
    request.body.resize(content_length);
    const std::size_t query = request.path.find('?');
    if (query != std::string::npos) request.path.resize(query);
    return request;
}

std::string http_response(const int code, const std::string& body, const std::string& origin = "") {
    const char* reason = code == 200 ? "OK" : code == 201 ? "Created" : code == 204 ? "No Content" :
                         code == 400 ? "Bad Request" : code == 401 ? "Unauthorized" :
                         code == 404 ? "Not Found" : code == 405 ? "Method Not Allowed" :
                         code == 409 ? "Conflict" : code == 413 ? "Payload Too Large" :
                         code == 422 ? "Unprocessable Content" : "Internal Server Error";
    std::ostringstream response;
    response << "HTTP/1.1 " << code << " " << reason << "\r\n"
             << "Content-Type: application/json; charset=utf-8\r\n"
             << "Content-Length: " << body.size() << "\r\n"
             << "Connection: close\r\n"
             << "X-Content-Type-Options: nosniff\r\n";
    if (!origin.empty()) {
        response << "Access-Control-Allow-Origin: " << origin << "\r\n"
                 << "Access-Control-Allow-Credentials: true\r\n"
                 << "Vary: Origin\r\n";
    }
    response << "\r\n" << body;
    return response.str();
}

bool constant_time_equal(const std::string& left, const std::string& right) {
    if (left.size() != right.size()) return false;
    unsigned char result = 0;
    for (std::size_t i = 0; i < left.size(); ++i) result |= static_cast<unsigned char>(left[i] ^ right[i]);
    return result == 0;
}

class Gateway {
public:
    Gateway(SessionStore& store, std::string api_key, std::string cors_origin)
        : store_(store), api_key_(std::move(api_key)), cors_origin_(std::move(cors_origin)) {}

    std::string handle(const Request& request) {
        const auto origin_it = request.headers.find("origin");
        const std::string allowed_origin = origin_it != request.headers.end() && origin_it->second == cors_origin_ ? cors_origin_ : "";
        if (request.method == "OPTIONS") {
            return "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Max-Age: 600\r\n" +
                   (allowed_origin.empty() ? "" : "Access-Control-Allow-Origin: " + allowed_origin + "\r\nAccess-Control-Allow-Credentials: true\r\n") + "Content-Length: 0\r\nConnection: close\r\n\r\n";
        }
        if (request.path == "/health" && request.method == "GET") {
            return http_response(200, "{\"status\":\"ok\",\"model\":\"native-session-context\",\"version\":\"" MNEMOSYNE_GATEWAY_VERSION "\"}", allowed_origin);
        }
        if (!authorized(request)) return http_response(401, "{\"detail\":\"invalid or missing bearer token\"}", allowed_origin);
        if (request.path == "/" && request.method == "GET") {
            return http_response(200, "{\"service\":\"MNEMOSYNE-MNS\",\"version\":\"" MNEMOSYNE_GATEWAY_VERSION "\",\"memory_mode\":\"session\",\"endpoints\":{\"health\":\"/health\",\"context_turn\":\"/context/turn\",\"context_recent\":\"/context/recent\",\"context_checkpoint\":\"/context/checkpoint\"}}", allowed_origin);
        }
        try {
            if (request.path == "/context/turn" && request.method == "POST") return record_turn(request, allowed_origin);
            if (request.path == "/context/checkpoint" && request.method == "POST") return record_checkpoint(request, allowed_origin);
            if (request.path == "/context/recent" && request.method == "POST") return recent(request, allowed_origin);
            if (request.path == "/retrieve" || request.path == "/retrieve_store" || request.path == "/route" || request.path == "/conflicts" || request.path == "/models" || request.path == "/detect_citations") {
                return http_response(409, "{\"detail\":\"endpoint is disabled in native session mode; use /context/recent instead\"}", allowed_origin);
            }
            return http_response(404, "{\"detail\":\"not found\"}", allowed_origin);
        } catch (const std::invalid_argument& error) {
            return http_response(422, "{\"detail\":\"" + json_escape(error.what()) + "\"}", allowed_origin);
        } catch (const std::exception& error) {
            std::cerr << "request error: " << error.what() << "\n";
            return http_response(500, "{\"detail\":\"internal server error\"}", allowed_origin);
        }
    }

private:
    bool authorized(const Request& request) const {
        if (api_key_.empty()) return true;
        const auto it = request.headers.find("authorization");
        const std::string expected = "Bearer " + api_key_;
        return it != request.headers.end() && constant_time_equal(it->second, expected);
    }

    static std::string required(const std::string& body, const std::string& key) {
        const auto value = json_string(body, key);
        if (!value.has_value() || value->empty()) throw std::invalid_argument(key + " is required");
        return *value;
    }

    std::string record_turn(const Request& request, const std::string& origin) {
        const std::string id = store_.record_turn(required(request.body, "session_id"), required(request.body, "turn_id"),
                                                  json_string(request.body, "project_id").value_or(""),
                                                  required(request.body, "user_text"), required(request.body, "assistant_text"));
        return http_response(200, "{\"memory_id\":\"" + json_escape(id) + "\"}", origin);
    }

    std::string record_checkpoint(const Request& request, const std::string& origin) {
        const std::string id = store_.record_checkpoint(required(request.body, "session_id"), required(request.body, "turn_id"),
                                                        json_string(request.body, "project_id").value_or(""), required(request.body, "transcript"));
        return http_response(200, "{\"memory_id\":\"" + json_escape(id) + "\"}", origin);
    }

    std::string recent(const Request& request, const std::string& origin) {
        const std::string session_id = required(request.body, "session_id");
        const int requested_max = json_int(request.body, "max_tokens").value_or(4000);
        if (requested_max < 100 || requested_max > 16000) throw std::invalid_argument("max_tokens must be between 100 and 16000");
        const auto items = store_.recent(session_id, json_string(request.body, "project_id").value_or(""), requested_max);
        int total = 0;
        std::ostringstream body;
        body << "{\"results\":[";
        for (std::size_t i = 0; i < items.size(); ++i) {
            const auto& item = items[i];
            if (i > 0) body << ',';
            body << "{\"memory_id\":\"" << json_escape(item.id) << "\",\"content\":\"" << json_escape(item.content)
                 << "\",\"created_at\":" << item.created_at << ",\"token_cost\":" << item.token_cost << '}';
            total += item.token_cost;
        }
        body << "],\"token_cost\":" << total << '}';
        return http_response(200, body.str(), origin);
    }

    SessionStore& store_;
    std::string api_key_;
    std::string cors_origin_;
};

void send_all(const SOCKET socket, const std::string& response) {
    std::size_t offset = 0;
    while (offset < response.size()) {
        const int written = send(socket, response.data() + offset, static_cast<int>(std::min<std::size_t>(response.size() - offset, 8192)), 0);
        if (written <= 0) return;
        offset += static_cast<std::size_t>(written);
    }
}

int parse_port(const std::string& text) {
    try {
        const int value = std::stoi(text);
        if (value < 1 || value > 65535) throw std::out_of_range("port");
        return value;
    } catch (...) { throw std::invalid_argument("port must be between 1 and 65535"); }
}

struct Options { std::string db_path = "data/memory.db"; std::string host = "127.0.0.1"; int port = 8000; };

Options parse_options(const int argc, char** argv) {
    Options options;
    if (const char* database = std::getenv("MNEMOSYNE_DB"); database != nullptr && *database != '\0') options.db_path = database;
    if (const char* port = std::getenv("MNEMOSYNE_PORT"); port != nullptr && *port != '\0') options.port = parse_port(port);
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--db" && i + 1 < argc) options.db_path = argv[++i];
        else if (argument == "--port" && i + 1 < argc) options.port = parse_port(argv[++i]);
        else if (argument == "--host" && i + 1 < argc) options.host = argv[++i];
        else if (argument == "--help") {
            std::cout << "Usage: mnemosyne_gateway [--db PATH] [--host 127.0.0.1] [--port 8000]\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown or incomplete option: " + argument);
    }
    if (options.host != "127.0.0.1" && options.host != "localhost") {
        throw std::invalid_argument("native gateway only supports loopback binding");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        WSADATA wsa{};
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) throw std::runtime_error("WSAStartup failed");
        SqliteApi sqlite;
        SessionStore store(sqlite, options.db_path);
        const char* key = std::getenv("MNEMOSYNE_API_KEY");
        const char* cors = std::getenv("MNEMOSYNE_CORS_ORIGIN");
        Gateway gateway(store, key == nullptr ? "" : key, cors == nullptr ? "http://127.0.0.1:3000" : cors);

        const SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listener == INVALID_SOCKET) throw std::runtime_error("could not create listening socket");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<u_short>(options.port));
        InetPtonA(AF_INET, "127.0.0.1", &address.sin_addr);
        if (bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR || listen(listener, SOMAXCONN) == SOCKET_ERROR) {
            closesocket(listener);
            throw std::runtime_error("could not bind native gateway to 127.0.0.1:" + std::to_string(options.port));
        }
        std::cout << "MNEMOSYNE native session gateway listening on http://127.0.0.1:" << options.port << " using " << options.db_path << "\n";
        for (;;) {
            const SOCKET client = accept(listener, nullptr, nullptr);
            if (client == INVALID_SOCKET) continue;
            std::thread([client, &gateway] {
                const auto request = read_request(client);
                send_all(client, request.has_value() ? gateway.handle(*request) : http_response(400, "{\"detail\":\"invalid HTTP request\"}"));
                closesocket(client);
            }).detach();
        }
    } catch (const std::exception& error) {
        std::cerr << "mnemosyne_gateway: " << error.what() << "\n";
        return 1;
    }
}
