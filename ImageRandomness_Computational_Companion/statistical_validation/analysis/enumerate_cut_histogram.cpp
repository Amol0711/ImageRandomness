// Exact integer histogram for weighted cuts of fully contained 2x2 windows.
// Build with C++17. The Python driver builds only inside a temporary directory.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: enumerate_cut_histogram ROWS COLS full|sparse MAX_K\n";
        return 2;
    }
    const int rows = std::stoi(argv[1]), cols = std::stoi(argv[2]);
    const std::string mode(argv[3]);
    const int n = rows * cols, maximum_k = std::stoi(argv[4]);
    if (rows < 2 || cols < 2 || n > 1024 || maximum_k < 0 ||
        (mode != "full" && mode != "sparse") ||
        (mode == "full" && (n > 20 || maximum_k != n)) ||
        (mode == "sparse" && maximum_k > 4)) {
        throw std::invalid_argument("unsupported enumeration dimensions or mode");
    }
    std::vector<int> weight(n*n, 0), degree(n, 0);
    // Accumulate all six unordered pairs separately in every complete window.
    for (int r=0; r<rows-1; ++r) for (int c=0; c<cols-1; ++c) {
        const int p[4] = {r*cols+c, r*cols+c+1, (r+1)*cols+c,
                          (r+1)*cols+c+1};
        for (int a=0; a<4; ++a) for (int b=a+1; b<4; ++b) {
            ++weight[p[a]*n+p[b]]; ++weight[p[b]*n+p[a]];
            ++degree[p[a]]; ++degree[p[b]];
        }
    }
    const int total_weight = 6*(rows-1)*(cols-1);
    std::vector<std::vector<std::uint64_t>> hist(
        maximum_k+1, std::vector<std::uint64_t>(total_weight+1, 0));
    auto add = [&](int k, int cut) {
        if (cut < 0 || cut > total_weight) throw std::logic_error("invalid cut");
        ++hist[k][cut];
    };
    if (mode == "full") {
        for (std::uint64_t mask=0; mask<(std::uint64_t(1)<<n); ++mask) {
            int k=0, cut=0;
            for (int i=0; i<n; ++i) {
                const bool occupied = (mask>>i)&1;
                k += occupied;
                for (int j=i+1; j<n; ++j)
                    if (occupied != bool((mask>>j)&1)) cut += weight[i*n+j];
            }
            add(k, cut);
        }
    } else {
        add(0, 0);
        if (maximum_k >= 1) for (int a=0; a<n; ++a) {
            const int ca=degree[a];
            add(1,ca);
            if (maximum_k >= 2) for (int b=a+1; b<n; ++b) {
                const int cab=ca+degree[b]-2*weight[a*n+b];
                add(2,cab);
                if (maximum_k >= 3) for (int c=b+1; c<n; ++c) {
                    const int cabc=cab+degree[c]-2*(weight[a*n+c]+weight[b*n+c]);
                    add(3,cabc);
                    if (maximum_k >= 4) for (int d=c+1; d<n; ++d) {
                        const int cabcd=cabc+degree[d]-2*(weight[a*n+d]+weight[b*n+d]+weight[c*n+d]);
                        add(4,cabcd);
                    }
                }
            }
        }
    }
    std::cout << "occupied,cut,count\n";
    for (int k=0; k<=maximum_k; ++k)
        for (int cut=0; cut<=total_weight; ++cut)
            if (hist[k][cut]) std::cout << k << ',' << cut << ',' << hist[k][cut] << '\n';
}
