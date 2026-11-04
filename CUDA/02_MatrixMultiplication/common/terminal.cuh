#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace terminal {

// ANSI styling is enabled for modern Windows Terminal / VS Code terminals.
// Call enable_ansi_colors() once at program startup.
namespace color {
    inline constexpr const char* reset  = "\033[0m";
    inline constexpr const char* bold   = "\033[1m";
    inline constexpr const char* dim    = "\033[2m";
    inline constexpr const char* red    = "\033[31m";
    inline constexpr const char* green  = "\033[32m";
    inline constexpr const char* yellow = "\033[33m";
    inline constexpr const char* blue   = "\033[34m";
    inline constexpr const char* cyan   = "\033[36m";
    inline constexpr const char* white  = "\033[37m";
}

inline bool enable_ansi_colors() {
#ifdef _WIN32
    HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (handle == INVALID_HANDLE_VALUE || handle == nullptr) {
        return false;
    }

    DWORD mode = 0;
    if (!GetConsoleMode(handle, &mode)) {
        // stdout may be redirected to a file or pipe.
        return false;
    }

    mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    return SetConsoleMode(handle, mode) != 0;
#else
    return true;
#endif
}

inline std::string styled(
    const std::string& text,
    const char* style,
    bool colors_enabled = true)
{
    if (!colors_enabled) {
        return text;
    }
    return std::string(style) + text + color::reset;
}

inline void print_banner(
    const std::string& title,
    const std::string& subtitle = "GPU KERNEL LAB",
    int width = 62)
{
    if (width < 20) width = 20;

    const std::string top = "+" + std::string(width - 2, '-') + "+";
    const std::string bottom = "+" + std::string(width - 2, '-') + "+";

    [[maybe_unused]] const auto centered = [width](const std::string& text) {
        const int inner = width - 2;
        const int left = (inner - static_cast<int>(text.size())) / 2;
        const int right = inner - static_cast<int>(text.size()) - left;
        return "│" + std::string(left > 0 ? left : 0, ' ') +
               text + std::string(right > 0 ? right : 0, ' ') + "│";
    };

    const auto centered_ascii = [width](const std::string& text) {
        const int inner = width - 2;
        const int left = (inner - static_cast<int>(text.size())) / 2;
        const int right = inner - static_cast<int>(text.size()) - left;
        return "|" + std::string(left > 0 ? left : 0, ' ') +
               text + std::string(right > 0 ? right : 0, ' ') + "|";
    };

    std::cout << color::cyan << top << '\n'
              << centered_ascii(subtitle) << '\n'
              << centered_ascii(title) << '\n'
              << bottom << color::reset << "\n\n";
}

inline void section(const std::string& title) {
    std::cout << color::bold << color::cyan
              << title << '\n'
              << color::reset
              << std::string(title.size(), '-')
              << '\n';
}

inline void key_value(
    const std::string& key,
    const std::string& value,
    int key_width = 24)
{
    std::cout << color::dim << std::left
              << std::setw(key_width) << key
              << color::reset
              << " : " << value << '\n';
}

inline void print_table(
    const std::vector<std::string>& headers,
    const std::vector<std::vector<std::string>>& rows)
{
    if (headers.empty()) return;

    std::vector<size_t> widths(headers.size(), 0);

    for (size_t col = 0; col < headers.size(); ++col) {
        widths[col] = headers[col].size();
    }

    for (const auto& row : rows) {
        for (size_t col = 0; col < headers.size() && col < row.size(); ++col) {
            widths[col] = std::max(widths[col], row[col].size());
        }
    }

    const auto border = [&]() {
        std::string line = "+";
        for (size_t width : widths) {
            line += std::string(width + 2, '-');
            line += "+";
        }
        return line;
    };

    const auto print_row = [&](const std::vector<std::string>& row,
                               bool header = false) {
        std::cout << "|";
        for (size_t col = 0; col < headers.size(); ++col) {
            const std::string value =
                col < row.size() ? row[col] : std::string();
            std::cout << ' ';
            if (header) std::cout << color::bold;
            std::cout << std::left << std::setw(static_cast<int>(widths[col]))
                      << value;
            if (header) std::cout << color::reset;
            std::cout << " |";
        }
        std::cout << '\n';
    };

    std::cout << border() << '\n';
    print_row(headers, true);
    std::cout << border() << '\n';
    for (const auto& row : rows) {
        print_row(row);
    }
    std::cout << border() << '\n';
}

inline void status(bool passed, const std::string& message = "TEST") {
    if (passed) {
        std::cout << color::green << color::bold
                  << "[PASS] " << color::reset
                  << message << '\n';
    } else {
        std::cout << color::red << color::bold
                  << "[FAIL] " << color::reset
                  << message << '\n';
    }
}

inline std::string format_float(
    double value,
    int precision = 3,
    bool scientific = false)
{
    std::ostringstream out;
    if (scientific) {
        out << std::scientific;
    } else {
        out << std::fixed;
    }
    out << std::setprecision(precision) << value;
    return out.str();
}

inline std::string format_shape(int rows, int cols) {
    return std::to_string(rows) + " x " + std::to_string(cols);
}

} // namespace terminal
