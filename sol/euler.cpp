/**
 * @file beam_search.cpp
 * @author titan23
 * @brief
 *  ルールベース+ビームサーチ
 * と思ったが、ルールベース取り除いた方が強い…？
 *
 * limit=5sec
    57
    DDDLLUULURDLDRURRDLLURURDDLUULDLDRRDLUURRDLLLDRULURRDLDRR
    104
    URDRRDLDLDLULURURRRULDDLDDRURUUULLDDDRDRUULULDDLLUURURDDRDLLDRRRULLLDLURDRUULLUURDLDDDRURRUURULLDDLDRRDR
    1108
    LULDDRULDDLDRURULLDDDDDRULURDRULURDRULURDRULURRDLUULDLURUUURDDLUULDDRULLDDLDDLUURURRDLURRRRDDDDDDLDLULLURULLDRRRRDLLLURURRDLUURDLULULULDRDLULLURURRRDRDRDRRUUULDDRURULLDLDLLLLLDLDDRRRULULLDDRRDRDLULLURURUULUUURDLLDRRURRULURDLLURDLLLDDRURRRRDRUULDDRUURDRURDLLLUULDDLULLLULDDRUURRRRDRURDRRULDDRUULDLLULLDDDDLLLLDDDDDRRULLDRULUURDLUURDLDRRRULURDRRRURDDRDRDLLURRULULDDLDLULUURRRDRDLULDLUURDDRURUUURUULULDDRULLULLDRRDRDRULDDLLLLDRRRRULULDDRUUULLURDDLUULULDLDDRDRURUULLDDDDLDDDRULURDRULURUURDDLURRDRDRUULDDDLURDDLUULURUURURDRULULLDRUULDLLDRRRDRUULLLLLDDRURDRRDLDRURUURRUULDLLLLLLDRDDLDRULURDDDDLUULDRUULDRUULDRURRRDDDLLLUUUULDRRRDLDRDDRDLLURRRDLLLLURRRDLLURUUURRRURURULLLDLULDDRUULDDRULLLDRDRUULUULDLDDRURUURDRDRRRULULDDDRUUURDDLULDLLLLLUURRDLLDRURDDLUURDDLDDRDLUURUULDRDDLDLDRDLURRULDDRULLDRUURUUURRRDRDDLLDRRRUURDLLLUURDRRULDDLDDLUUURURULLDLLDLUURDRDRURDDDRRDLURUULDDLLULLULUURRDRDRDLLLUURDRURDLURRDRUULDDDLUUUULLLDDRDDDLDLURDRULDLURULUUUURRDRURRDDRUULLDDRDRUULLLDDLLDRRURUURULLLLDRURDDLULLDDRRURURURDDLULLLDRURDDDLDRULUURDRDRRUULLLDDRRURDDLURDLLDLULURRRULLLDRRDLURRURULLDLDRURDLDDRUULURRDDLUULDDDLLURRDRRULLDRURD

 * limit=inf
    1006 40sec
    LULDRDLLLULURRDRRDLLLULDDRUURRDLDDLUUUURRURRULLDRRDDRULLDRDDLULDDLDDDLUUUUUURDDDDDRULLUUURDRDDRRURRDRURULURUULLDLDRURDLDDDRDLURRDLURULUULDLLLLDDRURRRDRUULLLLLDRRRRDLDLDLLUULLUURDRRRRULLUULUURRURRRDDLULLURRRDLLDDDLDLULUUULLDRRURRULLLLDDDDLUUUURDDDLUURDRDLDLUURDDLDRURURRRDRURRDRUUULLLDLLLLURRRRULDDDRRDRRUULDRDDLLDLULDDDRDLLLLLLURDRULUURRRRDLLLURRULLLDRUUULLDRDDDRDRRRULLULDRULUUULDRUUUURRRDDLULDRULLDRRDDDLULUURRDRRDRDLDDDRRDRULDLURUURULDLURUULULURURRDLDLUURDLULDLLDDRDDLLDRUULDRUULDRRRRDLDDLLLDDLLLURRRRRRRRULLLDDLUURDRULDLUUUURURDRRRULLUUULDLLLDDDRRRRUULLDLDRRDDRUUUULDLLDRRDLDDRURDDLURRRDDLULURULUUULDDRULLDDDRRDLDLLLLLLURURUUUUULDLDRRRRRULULUULLDDDRRULLDLUUURDLDRDRRRDDRRRRRDDLURDDLLDLLUULLDRDLLLURRULULUULDDDRDRRRURRDRUULURURDRULUULULLLULLDRRDDRURDDRRDDRUULULUUULDLULLLDLURRDRDLULURRRRRRRDDDLUUURDLULLDDRRRULULLDRURDLDRDDDLLLULDLLDRULURRDLDLLURULLDDRRRDRUULULULDLURRDDLLURDDLUURRDRRUURDDRRDRDLLUURDDRURULLLULURDLULURDDRULDRUULURRDDLUULDLDRRRUULDDDDDRDLLLDDRRULURRDRDLLLLURRURRUURDDDLDRULLDRRUULLLDRRDR




g++ tree.cpp -I./../../Library_cpp
time ./a.out < ./../in/testcase1.txt > out.txt
 */

// ログの出力

#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")

#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <cassert>
#include <chrono>
#include <thread>
#include <algorithm>

#include "titan_cpplib/algorithm/random.cpp"
#include "titan_cpplib/ahc/state_pool.cpp"
#include "titan_cpplib/ahc/timer.cpp"
#include "titan_cpplib/data_structures/hash_set.cpp"
#include "titan_cpplib/others/print.cpp"

using namespace std;

#define rep(i, n) for (int i = 0; i < (n); ++i)

template<typename T>
T abs(const T a, const T b) { return a > b ? a-b : b-a; }

int N;
int DAMMY;
vector<vector<int>> A;

std::string zfill(const int num, const int width) {
    if (num == DAMMY) return "-1";
    std::string str = std::to_string(num);
    return std::string(width - str.size(), '0') + str;
}

void input() {
    cin >> N;
    DAMMY = N*N;
    A.resize(N, vector<int>(N));
    rep(i, N) rep(j, N) {
        cin >> A[i][j];
    }
}

//! 木上のビームサーチライブラリ
namespace beam_search_with_tree {

using ScoreType = long long;
ScoreType INF = 1e9;
using HashType = unsigned long long;

// Action
struct Action {
    char d;
    ScoreType pre_score, nxt_score;
    HashType pre_hash, nxt_hash;

    Action() {}
    Action(const char d) : d(d), pre_score(INF), nxt_score(INF), pre_hash(0), nxt_hash(0) {}
};
ostream& operator<<(ostream& os, const Action &action) {
    os << action.d;
    return os;
}


HashType hash_rand[100][101];

// trans[i*N+j]:= (i,j)のときの遷移
vector<vector<Action>> trans;

void init_zhs() {
    titan23::Random my_rand;
    rep(i, N) rep(j, N) {
        rep(num, N*N+1) {
            hash_rand[i*N+j][num] = my_rand.rand_u64();
        }
    }

    trans.resize(N*N);
    rep(ij, N*N) {
        int i = ij/N, j = ij%N;
        if (i-1 >= 0) trans[ij].emplace_back('U');
        if (i+1 < N ) trans[ij].emplace_back('D');
        if (j-1 >= 0) trans[ij].emplace_back('L');
        if (j+1 < N ) trans[ij].emplace_back('R');
    }
}

class State {
  private:
    // (i, j) に val があるときのスコア
    int get_pos_d(const int i, const int j, const int val) const {
        if (val == 0) return 0;
        int s = abs(i-((val-1)/N)) + abs(j-((val-1)%N));
        return s;
    }

  public:
    vector<int> field;
    titan23::Random srand;
    ScoreType score;
    HashType hash;
    int zi, zj;

    State() {}

    // TODO Stateを初期化する
    void init() {
        field.resize(N*N);
        this->hash = 0;
        rep(i, N) rep(j, N) {
            const int a = A[i][j];
            int u = (a-1) / N;
            int v = (a-1) % N;
            this->field[i*N+j] = a;
            this->hash ^= hash_rand[i*N+j][a];
            if (a == 0) {
                zi = i;
                zj = j;
            }
        }
        this->score = 0;
        rep(i, N) rep(j, N) {
            this->score += get_pos_d(i, j, field[i*N+j]);
        }
    }

    // TODO
    //! `action` をしたときの評価値とハッシュ値を返す
    //! ロールバックに必要な情報はすべてactionにメモしておく
    pair<ScoreType, HashType> try_op(Action &action) const {
        // cerr << "try_op-start" << endl;
        int new_score = score;
        HashType new_hash = hash;

        action.pre_score = score;
        action.pre_hash = hash;

        // ni, nj
        int ni = zi, nj = zj;
        switch (action.d) {
            case 'D': ++ni; break;
            case 'R': ++nj; break;
            case 'U': --ni; break;
            case 'L': --nj; break;
        }

        // score
        new_score -= get_pos_d(zi, zj, field[zi*N+zj]);
        new_score += get_pos_d(zi, zj, field[ni*N+nj]);
        new_score -= get_pos_d(ni, nj, field[ni*N+nj]);
        new_score += get_pos_d(ni, nj, field[zi*N+zj]);

        // hash
        new_hash ^= hash_rand[zi*N+zj][field[zi*N+zj]];
        new_hash ^= hash_rand[zi*N+zj][field[ni*N+nj]];
        new_hash ^= hash_rand[ni*N+nj][field[ni*N+nj]];
        new_hash ^= hash_rand[ni*N+nj][field[zi*N+zj]];

        action.nxt_score = new_score;
        action.nxt_hash = new_hash;
        // cerr << "try_op-end" << endl;
        return {new_score, new_hash};
    }

    bool is_done() const { return this->get_score() == 0; }

    // TODO
    //! `action` をする
    void apply_op(const Action &action) {
        int ni = zi, nj = zj;
        switch (action.d) {
            case 'D': ++ni; break;
            case 'R': ++nj; break;
            case 'U': --ni; break;
            case 'L': --nj; break;
        }
        swap(field[zi*N+zj], field[ni*N+nj]);
        zi = ni;
        zj = nj;
        this->hash = action.nxt_hash;
        this->score = action.nxt_score;
    }

    // TODO
    //! `action` を戻す
    void rollback(const Action &action) {
        int ni = zi, nj = zj;
        switch (action.d) {
            case 'D': --ni; break;
            case 'R': --nj; break;
            case 'U': ++ni; break;
            case 'L': ++nj; break;
        }
        swap(field[zi*N+zj], field[ni*N+nj]);
        zi = ni;
        zj = nj;
        this->hash = action.pre_hash;
        this->score = action.pre_score;
    }

    // TODO
    //! 現状態から遷移可能な `Action` の `vector` を返す
    vector<Action> get_actions() const {
        assert(zi*N+zj < trans.size());
        return trans[zi*N+zj];
    }

    ScoreType get_score() const {
        return this->score;
    }

    void print() const {
        rep(i, N) {
            rep(j, N) cerr << zfill(field[i*N+j], 2) << ' ';
            cerr << endl;
        }
    }
};

struct BeamParam {
    int MAX_TURN;
    int BEAM_WIDTH;
};

class BeamSearchWithTree {
  private:
    titan23::HashSet seen;

    vector<tuple<int, int, Action>> tree; // dir, id, action
    vector<tuple<int, ScoreType, Action>> next_beam; // <par, score, action

    int get_next_beam(State* state, int turn) {
        next_beam.clear();
        next_beam.reserve(tree.size() * 4);
        seen.clear();

        if (turn == 0) {
            vector<Action> actions = state->get_actions();
            for (Action &action : actions) {
                auto [score, hash] = state->try_op(action);
                if (seen.contains_insert(hash)) continue;
                next_beam.emplace_back(-1, score, action);
            }
            return 0;
        }

        int cnt = 0;
        int leaf_id = 0;
        for (int i = 0; i < tree.size(); ++i) {
            auto [dir, _, action] = tree[i];
            if (dir >= 0) {
                state->apply_op(action);
                vector<Action> actions = state->get_actions();
                get<1>(tree[i]) = leaf_id;
                for (Action &action : actions) {
                    auto [score, hash] = state->try_op(action);
                    if (seen.contains_insert(hash)) continue;
                    next_beam.emplace_back(leaf_id, score, action);
                }
                leaf_id++;
                state->rollback(action);
            } else if (dir == -1) {
                state->apply_op(action);
            } else {
                state->rollback(action);
            }
        }
        return cnt;
    }

    //! 不要なNodeを削除し、木を更新する
    void update_tree(const int turn) {
        vector<tuple<int, int, Action>> new_tree;
        new_tree.reserve(tree.size());
        if (turn == 0) {
            for (auto [par, _, new_action] : next_beam) {
                assert(par == -1);
                new_tree.emplace_back(0, 0, new_action);
            }
            swap(tree, new_tree);
            return;
        }

        int beam_idx = 0;
        for (int i = 0; i < tree.size(); ++i) {
            auto [dir, leaf_id, action] = tree[i];
            if (dir >= 0) {
                if (beam_idx >= (int)next_beam.size()) continue;
                auto [par, _, new_action] = next_beam[beam_idx];
                if (par == leaf_id) {
                    new_tree.emplace_back(-1, -1, action);
                    while (par == leaf_id) {
                        new_tree.emplace_back(0, leaf_id, new_action);
                        beam_idx++;
                        tie(par, _, new_action) = next_beam[beam_idx];
                    }
                    new_tree.emplace_back(-2, -1, action);
                }
            } else if (dir == -1) {
                new_tree.emplace_back(-1, -1, action);
            } else {
                int pre_dir = get<0>(new_tree.back());
                if (pre_dir == -1) {
                    new_tree.pop_back();
                } else {
                    new_tree.emplace_back(-2, -1, action);
                }
            }
        }
        swap(tree, new_tree);
    }

    vector<Action> get_result() {
        int best_id = -1;
        ScoreType best_score = 0;
        for (auto [par, score, _] : next_beam) {
            if (best_id == -1 || score < best_score) {
                best_score = score;
                best_id = par;
            }
        }
        assert(best_id != -1);
        cerr << "best_id=" << best_id << endl;

        vector<Action> result;
        for (int i = 0; i < tree.size(); ++i) {
            auto [dir, laef_idx, action] = tree[i];
            if (dir >= 0) {
                if (best_id == laef_idx) {
                    result.emplace_back(action);
                    return result;
                }
            } else if (dir == -1) {
                result.emplace_back(action);
            } else {
                result.pop_back();
            }
        }
        assert(false);
    }

  public:
    vector<Action> search(const BeamParam &param, const bool verbose = false) {
        State* state = new State;
        state->init();

        this->seen = titan23::HashSet(param.BEAM_WIDTH * 4);

        int now_turn = 0;

        for (int turn = 0; turn < param.MAX_TURN; ++turn) {
            if (verbose) cerr << "# turn : " << turn+1 << " ";

            // 次のビーム候補を求める
            int apply_only_turn = get_next_beam(state, turn-now_turn);

            now_turn += apply_only_turn;
            assert(!next_beam.empty());

            // ビームを絞る // TODO 評価値が一致した場合、親の評価値も参考にするなど
            // vector<tuple<int, ScoreType, Action>> next_beam; // <par, score, action
            int beam_width = min(param.BEAM_WIDTH, (int)next_beam.size());
            nth_element(next_beam.begin(), next_beam.begin() + beam_width, next_beam.end(), [&] (const tuple<int, ScoreType, Action> &left, const tuple<int, ScoreType, Action> &right) {
                return std::get<1>(left) < std::get<1>(right);
            });

            tuple<int, ScoreType, Action> bests = *min_element(next_beam.begin(), next_beam.begin() + beam_width, [&] (const tuple<int, ScoreType, Action> &left, const tuple<int, ScoreType, Action> &right) {
                return std::get<1>(left) < std::get<1>(right);
            });
            cerr << "best_score = " << get<1>(bests) << endl;
            if (get<1>(bests) == 0) {
                cerr << "find valid solution." << endl;
                vector<Action> result = get_result();
                result.emplace_back(get<2>(bests));
                return result;
            }

            // 探索木の更新
            std::sort(next_beam.begin(), next_beam.begin() + beam_width, [&] (const tuple<int, ScoreType, Action> &left, const tuple<int, ScoreType, Action> &right) {
                return std::get<0>(left) < std::get<0>(right);
            });
            update_tree(turn);
        }

        // 答えを復元する
        vector<Action> result = get_result();
        return result;
    }
};
} // namespace beam_search


void solve() {
    beam_search_with_tree::BeamParam param;
    param.MAX_TURN = 60;
    param.BEAM_WIDTH = 100000;
    beam_search_with_tree::init_zhs();
    beam_search_with_tree::BeamSearchWithTree bs;
    vector<beam_search_with_tree::Action> ans = bs.search(param, true);
    for (const beam_search_with_tree::Action &action : ans) {
        cout << action;
    }
    cout << endl;

    cerr << "Score = " << ans.size() << endl;
}

int main() {
    input();
    solve();
    return 0;
}
