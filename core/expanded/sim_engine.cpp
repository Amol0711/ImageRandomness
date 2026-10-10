// Deterministic BACI numerator evaluator and independent-uniform simulator.
// C++17; PRNG is standardized std::mt19937_64, with explicit byte extraction.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

static uint64_t number(const char* text) {
    std::string s(text);
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected an unsigned decimal integer");
    size_t pos = 0;
    uint64_t v = std::stoull(s, &pos, 10);
    if (pos != s.size()) throw std::invalid_argument("invalid integer suffix");
    return v;
}

static uint64_t splitmix64(uint64_t x) {
    // All arithmetic is modulo 2^64; this is a bijection of 64-bit inputs.
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static void dimensions(uint64_t m, uint64_t n) {
    if (m < 2 || n < 2 || m > 100000 || n > 100000)
        throw std::invalid_argument("dimensions must be in [2, 100000]");
    if (UINT64_C(6) * (m-1) * (n-1) * 255 > UINT32_MAX)
        throw std::invalid_argument("conservative per-image numerator exceeds uint32");
}

static std::vector<int16_t> read_image(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != static_cast<std::streamoff>(count))
        throw std::runtime_error("input must contain exactly rows*columns uint8 values");
    f.seekg(0);
    std::vector<uint8_t> bytes(count);
    if (!f.read(reinterpret_cast<char*>(bytes.data()), count))
        throw std::runtime_error("failed reading image");
    return std::vector<int16_t>(bytes.begin(), bytes.end());
}

static uint32_t numerator(const int16_t* a, size_t m, size_t n) {
    // Four edge orientations, counted with their literal-window multiplicities.
    // Horizontal edges occur once in boundary rows and twice in interior rows.
    // Vertical edges occur once in boundary columns and twice inside.
    // Each of the two diagonals occurs in exactly one 2x2 window.
    uint64_t total = 0;
    for (size_t row = 0; row < m; ++row) {
        const int16_t* current = a + row*n;
        uint64_t horizontal = 0;
        #pragma omp simd reduction(+:horizontal)
        for (size_t j=0; j<n-1; ++j)
            horizontal += static_cast<unsigned>(std::abs(int(current[j+1])-int(current[j])));
        total += ((row == 0 || row == m-1) ? 1 : 2)*horizontal;
        if (row != 0) {
            const int16_t* previous = current-n;
            uint64_t vertical_interior = 0, diagonal = 0;
            #pragma omp simd reduction(+:vertical_interior)
            for (size_t j=1; j<n-1; ++j)
                vertical_interior += static_cast<unsigned>(std::abs(int(current[j])-int(previous[j])));
            #pragma omp simd reduction(+:diagonal)
            for (size_t j=0; j<n-1; ++j)
                diagonal += static_cast<unsigned>(std::abs(int(current[j+1])-int(previous[j])))
                          + static_cast<unsigned>(std::abs(int(current[j])-int(previous[j+1])));
            total += 2*vertical_interior + diagonal
                   + static_cast<unsigned>(std::abs(int(current[0])-int(previous[0])))
                   + static_cast<unsigned>(std::abs(int(current[n-1])-int(previous[n-1])));
        }
    }
    return static_cast<uint32_t>(total);
}

static void draw_iid(std::mt19937_64& rng, std::vector<int16_t>& image) {
    size_t pos = 0;
    while (pos < image.size()) {
        uint64_t word = rng();
        // Low 16-bit chunk first. U is its low byte, V its high byte.
        for (unsigned k=0; k<4 && pos<image.size(); ++k, ++pos) {
            image[pos] = static_cast<int16_t>(std::abs(int(word & 255)-int((word >> 8) & 255)));
            word >>= 16;
        }
    }
    // Unused high bytes, if any, are discarded at each image boundary.
}

static void draw_fixed(std::mt19937_64& rng, std::vector<int16_t>& image,
                       const std::vector<int16_t>& fixed) {
    size_t pos = 0;
    while (pos < image.size()) {
        uint64_t word = rng();
        // Uniform comparison bytes are extracted low byte first.
        for (unsigned k=0; k<8 && pos<image.size(); ++k, ++pos) {
            image[pos] = static_cast<int16_t>(std::abs(int(fixed[pos])-int(word & 255)));
            word >>= 8;
        }
    }
}

static void write_u32_le(const std::string& path, const std::vector<uint32_t>& values) {
    // Explicit little endian, independent of the host's byte order.
    if (std::filesystem::exists(path)) throw std::runtime_error("output already exists");
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot create output file");
    for (uint32_t x : values) {
        const char bytes[4] = {static_cast<char>(x), static_cast<char>(x >> 8),
                               static_cast<char>(x >> 16), static_cast<char>(x >> 24)};
        f.write(bytes, 4);
    }
    if (!f) throw std::runtime_error("failed writing output file");
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::invalid_argument("modes: stat, benchmark, simulate");
        const std::string mode(argv[1]);
        if (mode == "stat") {
            if (argc != 5) throw std::invalid_argument("stat ROWS COLS IMAGE_UINT8");
            const size_t m=number(argv[2]), n=number(argv[3]);
            dimensions(m,n);
            const auto image=read_image(argv[4],m*n);
            std::cout << "{\"raw_numerator\":" << numerator(image.data(),m,n)
                      << ",\"scale\":" << UINT64_C(6)*(m-1)*(n-1)*255 << "}\n";
        } else if (mode == "benchmark") {
            if (argc != 5) throw std::invalid_argument("benchmark ROWS COLS ITERATIONS");
            const size_t m=number(argv[2]), n=number(argv[3]);
            const uint64_t iterations=number(argv[4]);
            dimensions(m,n);
            if (!iterations) throw std::invalid_argument("iterations must be positive");
            std::vector<int16_t> image(m*n);
            uint64_t checksum=0;
            const auto begin=std::chrono::steady_clock::now();
            for (uint64_t r=0; r<iterations; ++r) {
                // Entirely deterministic patterns, no stochastic observations.
                for (size_t i=0;i<image.size();++i)
                    image[i]=static_cast<int16_t>((i*73+(i/n)*17+r*19)&255);
                checksum += numerator(image.data(),m,n);
            }
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            std::cout << "{\"iterations\":" << iterations << ",\"checksum\":" << checksum
                      << ",\"elapsed_seconds\":" << elapsed << "}\n";
        } else if (mode == "simulate") {
            if (argc != 12) throw std::invalid_argument(
                "simulate ROWS COLS REPEATS START COUNT SEED_BASE iid|fixed FIXED_FILE|- OUTPUT THREADS");
            const size_t m=number(argv[2]), n=number(argv[3]);
            const uint64_t repeats=number(argv[4]), start=number(argv[5]), count=number(argv[6]), seed_base=number(argv[7]);
            const std::string model(argv[8]), fixed_path(argv[9]), output(argv[10]);
            const uint64_t threads=number(argv[11]);
            dimensions(m,n);
            if (!repeats || repeats>100000 || !count || count>10000000 || !threads || threads>1024)
                throw std::invalid_argument("invalid repeat, trial, or thread count");
            if (model != "iid" && model != "fixed") throw std::invalid_argument("model must be iid or fixed");
            if (start>UINT64_MAX-count || seed_base>UINT64_MAX-(start+count-1))
                throw std::invalid_argument("trial seed range would wrap");
            if (count>std::numeric_limits<size_t>::max()/repeats || count*repeats>1000000000)
                throw std::invalid_argument("requested output exceeds one billion images");
            std::vector<int16_t> fixed;
            if (model=="fixed") fixed=read_image(fixed_path,m*n);
            else if (fixed_path!="-") throw std::invalid_argument("iid model requires '-' for fixed file");
            std::vector<uint32_t> values(count*repeats);
            const auto begin=std::chrono::steady_clock::now();
            #pragma omp parallel num_threads(static_cast<int>(threads))
            {
                std::vector<int16_t> image(m*n);
                #pragma omp for schedule(static)
                for (uint64_t t=0; t<count; ++t) {
                    std::mt19937_64 rng(splitmix64(seed_base+start+t));
                    for (uint64_t r=0; r<repeats; ++r) {
                        if (model=="iid") draw_iid(rng,image);
                        else draw_fixed(rng,image,fixed);
                        values[t*repeats+r]=numerator(image.data(),m,n);
                    }
                }
            }
            write_u32_le(output,values);
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            uint64_t checksum=0;
            for (uint32_t x:values) checksum+=x;
            std::cout << "{\"rows\":" << m << ",\"columns\":" << n
                      << ",\"repeats\":" << repeats << ",\"start\":" << start
                      << ",\"trials\":" << count << ",\"seed_base\":" << seed_base
                      << ",\"threads\":" << threads << ",\"images\":" << values.size()
                      << ",\"checksum\":" << checksum << ",\"elapsed_seconds\":" << elapsed << "}\n";
        } else throw std::invalid_argument("unknown mode");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sim_engine: " << e.what() << '\n';
        return 2;
    }
}
