// Lotto649 Strategy Verification v1.5
// Exact-integer exhaustive search of the SHA-audited, primary-optimal M0 slice.
// This standalone enumerator intentionally does NOT invoke an MILP/LP solver.
// Four nonoverlapping six-number tickets, one in each of four shape partitions.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct Ticket {
    int id, partition;
    std::array<int,6> numbers;
    std::array<int,4> ranks;
    uint64_t mask;
};

void output_four_ints(const std::array<int,4>& x) {
    std::cout << '[' << x[0] << ',' << x[1] << ',' << x[2] << ',' << x[3] << ']';
}
int main() {
    try {
        int n, bound;
        if (!(std::cin >> n >> bound) || n < 4 || n > 50000 || bound < 0)
            throw std::runtime_error("invalid size/primary bound header");
        std::map<int,std::vector<Ticket>> groups;
        std::set<int> unique_ids;
        std::set<std::array<int,6>> unique_tickets;
        for (int i = 0; i < n; ++i) {
            Ticket t{};
            if (!(std::cin >> t.id >> t.partition)) throw std::runtime_error("incomplete row header");
            for (int &k : t.numbers) if (!(std::cin >> k)) throw std::runtime_error("missing ball number");
            for (int &k : t.ranks) if (!(std::cin >> k)) throw std::runtime_error("missing rank number");
            if (t.id != i || t.partition < 0 ||
                !unique_ids.insert(t.id).second || !unique_tickets.insert(t.numbers).second)
                throw std::runtime_error("invalid identifier or duplicate ticket");
            t.mask = 0;
            for (int j = 0; j < 6; ++j) {
                if (t.numbers[j] < 1 || t.numbers[j] > 49 ||
                    (j > 0 && t.numbers[j] <= t.numbers[j-1]))
                    throw std::runtime_error("ball numbers must be sorted, unique and legal");
                t.mask |= uint64_t{1} << (t.numbers[j]-1);
            }
            for (int k : t.ranks) if (k < 0) throw std::runtime_error("negative rank");
            groups[t.partition].push_back(t);
        }
        std::string extra;
        if (std::cin >> extra) throw std::runtime_error("unexpected trailing data");
        if (groups.size() != 4) throw std::runtime_error("exactly four shape partitions required");
        std::array<const std::vector<Ticket>*,4> g{};
        std::array<int,4> caps{};
        int shape_total = 0;
        int k = 0;
        uint64_t cartesian = 1;
        for (const auto &kv : groups) {
            const auto &v = kv.second;
            const int cap = v.front().ranks[3];
            for (const Ticket& t : v) if (t.ranks[3] != cap)
                throw std::runtime_error("reduced partition shape ranks are not equal");
            caps[k] = cap;
            shape_total += cap;
            cartesian *= static_cast<uint64_t>(v.size());
            g[k++] = &v;
        }
        if (shape_total != bound) throw std::runtime_error("shape upper-bound mismatch");
        uint64_t pairs = 0, triples = 0, quadruples = 0, primary = 0, tie_count = 0;
        int best_total = -1, best_joint = -1;
        std::array<int,4> best_ids{{-1,-1,-1,-1}};
        std::array<int,4> best_blocks{};
        for (const Ticket& a : *g[0]) {
          for (const Ticket& b : *g[1]) {
            if ((a.mask & b.mask) != 0) continue;
            ++pairs;
            const uint64_t mask_ab = a.mask | b.mask;
            const int s_ab = a.ranks[0] + b.ranks[0];
            const int l_ab = a.ranks[1] + b.ranks[1];
            const int j_ab = a.ranks[2] + b.ranks[2];
            for (const Ticket& c : *g[2]) {
              if ((mask_ab & c.mask) != 0) continue;
              ++triples;
              const uint64_t mask_abc = mask_ab | c.mask;
              const int s_abc = s_ab + c.ranks[0];
              const int l_abc = l_ab + c.ranks[1];
              const int j_abc = j_ab + c.ranks[2];
              for (const Ticket& d : *g[3]) {
                if ((mask_abc & d.mask) != 0) continue;
                ++quadruples;
                const int s = s_abc + d.ranks[0];
                const int l = l_abc + d.ranks[1];
                const int j = j_abc + d.ranks[2];
                if (s < bound || l < bound || j < bound) continue;
                ++primary;
                const int total = s + l + j + bound;
                const std::array<int,4> ids{{a.id,b.id,c.id,d.id}};
                if (total > best_total || (total == best_total && j > best_joint)) {
                    best_total = total;
                    best_joint = j;
                    best_ids = ids;
                    best_blocks = {{s,l,j,bound}};
                    tie_count = 1;
                } else if (total == best_total && j == best_joint) {
                    ++tie_count;
                    if (ids < best_ids) {
                        best_ids = ids;
                        best_blocks = {{s,l,j,bound}};
                    }
                }
              }
            }
          }
        }
        if (primary == 0 || best_total < 0) throw std::runtime_error("no primary-optimal legal portfolio");
        std::cout << "{\"schema\":\"lotto649.v15.exhaustive_int.v1\""
                  << ",\"candidates\":" << n << ",\"primary_bound\":" << bound
                  << ",\"cartesian_count\":" << cartesian
                  << ",\"disjoint_pairs\":" << pairs
                  << ",\"disjoint_triples\":" << triples
                  << ",\"disjoint_four_portfolios\":" << quadruples
                  << ",\"primary_feasible_portfolios\":" << primary
                  << ",\"tied_best_lexicographic_portfolios\":" << tie_count
                  << ",\"best_total_rank\":" << best_total
                  << ",\"best_joint_rank\":" << best_joint
                  << ",\"selected_candidate_ids\":";
        output_four_ints(best_ids);
        std::cout << ",\"best_rank_sums\":";
        output_four_ints(best_blocks);
        std::cout << ",\"partition_sizes\":[" << g[0]->size() << ',' << g[1]->size() << ',' << g[2]->size() << ',' << g[3]->size() << "]}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "EXHAUSTIVE_AUDIT_FAIL: " << e.what() << '\n';
        return 2;
    }
}
