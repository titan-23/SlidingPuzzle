/**
 * @file beam_search.cpp
 * @author titan23
 * @brief
 *  ルールベース+ビームサーチ
 * と思ったが、ルールベース取り除いた方が強い…？
 * 

g++ ./sol/beam_search.cpp -I./../../Library_cpp -O2

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
 */

// ログの出力
const bool LOGS = true;

#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")

#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <set>
#include <cassert>
#include <chrono>
#include <thread>
#include <algorithm>

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/hash_policy.hpp>

// #include "./random.cpp"
// #include "./util.cpp"
// #include "./action.cpp"
// #include "./state_pool.cpp"
// #include "./timer.cpp"
#include "titan_cpplib/others/print.cpp"
// #include "titan_cpplib/algorithm/random.cpp"
#include "titan_cpplib/ahc/state_pool.cpp"
#include "titan_cpplib/ahc/timer.cpp"
#include "titan_cpplib/data_structures/hash_set.cpp"

using namespace std;

// Action
namespace titan23 {

  enum class Action { U, R, D, L };
  ostream& operator<<(ostream& os, const Action &action) {
    switch (action) {
      case Action::U: os << 'U'; break;
      case Action::R: os << 'R'; break;
      case Action::D: os << 'D'; break;
      case Action::L: os << 'L'; break;
      default: assert(false);
    }
    return os;
  }

  Action get_rev_action(const Action &action) {
    switch (action) {
      case Action::U: return Action::D;
      case Action::D: return Action::U;
      case Action::R: return Action::L;
      case Action::L: return Action::R;
    }
    assert(false);
  }
}

// Random
namespace titan23 {

  struct Random {

   private:
    unsigned int _x, _y, _z, _w;

    unsigned int _xor128() {
      const unsigned int t = _x ^ (_x << 11);
      _x = _y;
      _y = _z;
      _z = _w;
      _w = (_w ^ (_w >> 19)) ^ (t ^ (t >> 8));
      return _w;
    }

   public:
    Random() : _x(123456789),
               _y(362436069),
               _z(521288629),
               _w(88675123) {}

    double random() { return (double)(_xor128()) / 0xFFFFFFFF; }

    int randint(const int end) {
      assert(0 <= end);
      return (((unsigned long long)_xor128() * (end+1)) >> 32);
    }

    int randint(const int begin, const int end) {
      assert(begin <= end);
      return begin + (((unsigned long long)_xor128() * (end-begin+1)) >> 32);
    }

    unsigned long long randrand() {
      return (unsigned long long)_xor128() * (unsigned long long)_xor128();
    }

    int randrange(const int end) {
      assert(0 < end);
      return (((unsigned long long)_xor128() * end) >> 32);
    }

    int randrange(const int begin, const int end) {
      assert(begin < end);
      return begin + (((unsigned long long)_xor128() * (end-begin)) >> 32);
    }

    double randdouble(const double begin, const double end) {
      assert(begin < end);
      return begin + random() * (end-begin);
    }

    template <typename T>
    void shuffle(vector<T> &a) {
      int n = (int)a.size();
      for (int i = 0; i < n-1; ++i) {
        int j = randrange(i, n);
        swap(a[i], a[j]);
      }
    }
  };
} // namespace titan23

#define rep(i, n) for (int i = 0; i < (n); ++i)

template<typename T>
T abs(const T a, const T b) { return a > b ? a-b : b-a; }

int N;
int DAMMY;
int GLOBAL_DO_UB;
vector<short> is_active;
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
  is_active.resize(N*N, 1);
}

// puzzle15
namespace titan23 {

namespace puzzle15 {
  struct State;
  using StatePtr = State*;
  using ScoreType = int;
  using HashType = unsigned long long;
  struct SubState;
  HashType hash_rand[100][101];
  StatePool<State> pool;
  StatePool<SubState> sub_pool;

  // trans[i*N+j][action]:= (i,j),last=actionのときの遷移
  vector<vector<vector<titan23::Action>>> trans;

  void init() {
    titan23::Random r;
    rep(i, N) rep(j, N) {
      rep(num, N*N+1) {
        hash_rand[i*N+j][num] = r.randrand();
      }
    }

    trans.clear();
    trans.resize(N*N, vector<vector<titan23::Action>>(5));
    rep(ij, N*N) rep(k, 5) {
      int i = ij/N, j = ij%N;
      if (i-1 >= 0 && is_active[(i-1)*N+j]) trans[ij][k].emplace_back(titan23::Action::U);
      if (i+1 < N  && is_active[(i+1)*N+j]) trans[ij][k].emplace_back(titan23::Action::D);
      if (j-1 >= 0 && is_active[i*N+(j-1)]) trans[ij][k].emplace_back(titan23::Action::L);
      if (j+1 < N  && is_active[i*N+(j+1)]) trans[ij][k].emplace_back(titan23::Action::R);
      if (k < 4) {
        titan23::Action rev = get_rev_action(static_cast<titan23::Action>(k));
        bool f = false;
        for (const Action &a: trans[ij][k]) {
          if (a == rev) {
            f = true;
            break;
          }
        }
        if (f) trans[ij][k].erase(remove(trans[ij][k].begin(), trans[ij][k].end(), rev), trans[ij][k].end());
      }
    }
  }

  class State {
   private:
    int man_dist;
    int zi, zj;

    // (i, j) に val があるときのスコア
    int get_pos_d(const int i, const int j, const int val) const {
      if (val == 0 || val == DAMMY) return 0;
      int s = abs(i-((val-1)/N)) + abs(j-((val-1)%N));
      return s*s;
    }

   public:
    vector<int8_t> field;
    HashType hash;
    Action first_action, last_action;
    int substate_id;
    State() {}

    bool is_done() const { return get_score() == 0; }

    void apply_op(const Action &action) {
      last_action = action;

      tie(man_dist, hash) = try_op(action);

      // ni, nj
      assert(field[zi*N+zj] == 0);
      int ni = zi, nj = zj;
      switch (action) {
        case Action::D: ++ni; break;
        case Action::R: ++nj; break;
        case Action::U: --ni; break;
        case Action::L: --nj; break;
      }
      swap(field[zi*N+zj], field[ni*N+nj]);
      zi = ni;
      zj = nj;
    }

    pair<int, HashType> try_op(const Action &action) const {
      int new_man_dist = man_dist;
      HashType new_hash = hash;

      // ni, nj
      int ni = zi, nj = zj;
      switch (action) {
        case Action::D: ++ni; break;
        case Action::R: ++nj; break;
        case Action::U: --ni; break;
        case Action::L: --nj; break;
      }

      // man_dist
      new_man_dist -= get_pos_d(zi, zj, field[zi*N+zj]);
      new_man_dist += get_pos_d(zi, zj, field[ni*N+nj]);
      new_man_dist -= get_pos_d(ni, nj, field[ni*N+nj]);
      new_man_dist += get_pos_d(ni, nj, field[zi*N+zj]);

      // hash
      new_hash ^= hash_rand[zi*N+zj][field[zi*N+zj]];
      new_hash ^= hash_rand[zi*N+zj][field[ni*N+nj]];
      new_hash ^= hash_rand[ni*N+nj][field[ni*N+nj]];
      new_hash ^= hash_rand[ni*N+nj][field[zi*N+zj]];

      return {new_man_dist, new_hash};
    }

    int get_score() const { return man_dist; }

    void print() const {
      rep(i, N) {
        rep(j, N) cerr << zfill(field[i*N+j], 2) << ' ';
        cerr << endl;
      }
    }

    vector<Action>& get_actions(const bool is_first_turn) const {
      if (is_first_turn) {
        return trans[zi*N+zj][4];
      }
      assert(0 <= zi && 0 <= zj);
      return trans[zi*N+zj][static_cast<int>(last_action)];
    }

    void init() {
      field.resize(N*N);
      hash = 0;
      rep(i, N) rep(j, N) {
        const int a = A[i][j];
        int u = (a-1) / N;
        int v = (a-1) % N;
        if (a==0 || u<GLOBAL_DO_UB || v<GLOBAL_DO_UB) {
          field[i*N+j] = a;
        } else {
          field[i*N+j] = DAMMY;
        }
        hash ^= hash_rand[i*N+j][a];
        if (a == 0) {
          zi = i;
          zj = j;
        }
      }
      man_dist = 0;
      rep(i, N) rep(j, N) {
        man_dist += get_pos_d(i, j, field[i*N+j]);
      }
    }

    void copy(StatePtr &new_state) const {
      new_state->first_action = first_action;
      new_state->last_action = last_action;
      new_state->man_dist = man_dist;
      new_state->field = field;
      new_state->hash = hash;
      new_state->zi = zi;
      new_state->zj = zj;
    }

    bool operator>(const State &right) const { return get_score() > right.get_score(); }
    bool operator<(const State &right) const { return get_score() < right.get_score(); }
  };

  struct Result {
    vector<Action> history;
    int score;
    Result() {}

    void print(const string end) const {
      for (Action action: history) cout << action;
      cout << end;
    }

    void print() const {
      for (Action action: history) cout << action;
      cout << endl;
    }
  };

  class Param {
   private:
    Timer timer;
    double time_limit;
    int beam_depth_base, beam_width_base;
    int decide_turn, max_turn;
    vector<int> pred, pred_acc;
    int done_depth_total;
    bool adjace;

   public:
    Param() {}

    /**
     *
     @brief Construct a new Param object
     *
     * @param time_limit
     * @param max_turn
     * @param beam_depth_base
     * @param beam_width_base
     */
    Param(const double time_limit,
          const int max_turn,
          const int decide_turn,
          const int beam_depth_base,
          const int beam_width_base,
          const bool adjace=false) :
        time_limit(time_limit),
        beam_depth_base(beam_depth_base), beam_width_base(beam_width_base),
        decide_turn(decide_turn),
        max_turn(max_turn),
        pred(max_turn+1, 0), pred_acc(max_turn+2, 0),
        done_depth_total(0),
        adjace(adjace) {
      vector<int> t(max_turn+1);
      for (int turn = 0; turn < max_turn+1; ++turn) {
        pred[turn] = max(1, min(beam_depth_base, max_turn-turn));
        pred_acc[turn+1] = pred_acc[turn] + pred[turn];
      }
    }

    void init() { timer.reset(); }
    int get_max_turn() const { return max_turn; }
    int get_beam_depth() const { return beam_depth_base; }
    int get_beam_width() const { return beam_width_base; }

    int get_beam_depth(const int turn) const {
      return min(beam_depth_base, max_turn-turn);
    }

    void timestamp(const int turn, const int done_depth) {
      done_depth_total += done_depth;
    }

    int get_decide_turn() const {
      return decide_turn;
    }

    int get_beam_width(const int turn) {
      if ((!adjace) || turn == 0) return get_beam_width();
      double now_time = timer.elapsed();
      double rem_time = time_limit - now_time;
      int t = pred_acc.back() - pred_acc[turn];
      double pred_total_cnt = rem_time * (double)done_depth_total / now_time;
      int d = max(1, (int)(pred[turn] * pred_total_cnt / (double)t));
      return d;
    }
  };

  struct SubState {
    ScoreType score;
    long long state, par;
    Action action;
    SubState() {}
  };

  class BeamSearch {
   private:
    static inline void calc_next_beam(const vector<long long> &keep,
                                      const int turn,
                                      __gnu_pbds::gp_hash_table<HashType, uint8_t> &seen,
                                      vector<long long> &score_keep) {
      for (const long long now_state: keep) {
        const vector<Action> &actions = pool.get(now_state)->get_actions(turn);
        for (const Action &op : actions) {
          auto [new_score, new_hash] = pool.get(now_state)->try_op(op);
          if (seen.find(new_hash) != seen.end()) continue;
          seen[new_hash] = 0;
          const long long substate = sub_pool.gen();
          sub_pool.get(substate)->score = new_score;
          sub_pool.get(substate)->state = now_state;
          sub_pool.get(substate)->action = op;
          score_keep.emplace_back(substate);
        }
      }
    }
   public:
    static inline Result run_each_turn(Param &param, const bool verbose=false) {
      Result result;

      const long long best_state = pool.gen();
      pool.get(best_state)->init();
      param.init();

      for (int turn = 0; turn < param.get_max_turn(); ++turn) {
        if (verbose) cout << "# turn : " << turn << endl;
        const int beam_depth = param.get_beam_depth(turn);
        const int beam_width = param.get_beam_width(turn);
        int done_depth = 0;
        vector<long long> keep = { pool.copy(best_state) };
        __gnu_pbds::gp_hash_table<HashType, uint8_t> seen;
        for (int beam_turn = 0; beam_turn < beam_depth; ++beam_turn, ++done_depth) {
          vector<long long> score_keep;
          score_keep.reserve(keep.size());
          calc_next_beam(keep, turn, seen, score_keep);
          nth_element(score_keep.begin(), score_keep.begin() + min((long long)score_keep.size(), (long long)param.get_beam_width()), score_keep.end(), [&] (const long long &l, const long long &r) {
            return sub_pool.get(l)->score < sub_pool.get(r)->score;
          });
          const Action &op = sub_pool.get(score_keep.front())->action;
          bool is_all_same = true;
          vector<long long> new_keep(min((long long)beam_width, (long long)score_keep.size()));
          for (int i = 0; i < beam_width && i < score_keep.size(); ++i) {
            const long long state = pool.copy(sub_pool.get(score_keep[i])->state);
            if (beam_turn == 0) {
              pool.get(state)->first_action = sub_pool.get(score_keep[i])->action;
            }
            pool.get(state)->apply_op(sub_pool.get(score_keep[i])->action);
            if (is_all_same && pool.get(state)->first_action != op) is_all_same = false;
            new_keep[i] = state;
          }
          for (const long long state: score_keep) sub_pool.del(state);
          for (const long long state: keep) pool.del(state);
          swap(keep, new_keep);
          const long long d_best_state = *min_element(keep.begin(), keep.end(), [&] (const long long &l, const long long &r) {
            return (*pool.get(l)) < (*pool.get(r));
          });
          if (pool.get(d_best_state)->is_done() || is_all_same) break;
        }
        param.timestamp(turn, done_depth);
        const long long d_best_state = *min_element(keep.begin(), keep.end(), [&] (const long long &l, const long long &r) {
          return (*pool.get(l)) < (*pool.get(r));
        });
        const Action &op = pool.get(d_best_state)->first_action;
        pool.get(best_state)->apply_op(op);
        result.history.push_back(op);
        if (verbose) {
          pool.get(best_state)->print();
          cout << "Score = " << pool.get(best_state)->get_score() << endl << endl;
        }
        for (const long long node_id: keep) pool.del(node_id);
        if (pool.get(best_state)->is_done()) break;
      }
      result.score = pool.get(best_state)->get_score();
      pool.del(best_state);
      return result;
    }

    static inline Result run_normal(Param &param, const bool verbose=false) {
      param.init();
      Result result;

      __gnu_pbds::gp_hash_table<HashType, uint8_t> seen;
      vector<long long> keep;

      { // init state
        const long long state = pool.gen();
        pool.get(state)->init();
        seen[state] = 0;
        keep = { state };
        pool.get(state)->substate_id = -1;
      }

      for (int turn = 0; turn < param.get_max_turn(); ++turn) {
        if (verbose) cout << "# turn : " << turn << endl;
        const int beam_width = param.get_beam_width(turn);
        vector<long long> score_keep;
        calc_next_beam(keep, turn, seen, score_keep);
        nth_element(score_keep.begin(), score_keep.begin() + min((long long)score_keep.size(), (long long)param.get_beam_width()), score_keep.end(), [&] (const long long &l, const long long &r) {
          return sub_pool.get(l)->score < sub_pool.get(r)->score;
        });
        vector<long long> new_keep(min((long long)beam_width, (long long)score_keep.size()));
        for (int i = 0; i < beam_width && i < score_keep.size(); ++i) {
          long long substate = score_keep[i];
          long long new_state = pool.copy(sub_pool.get(substate)->state);
          pool.get(new_state)->apply_op(sub_pool.get(substate)->action);
          pool.get(new_state)->substate_id = substate;
          sub_pool.get(substate)->par = pool.get(sub_pool.get(substate)->state)->substate_id;
          new_keep[i] = new_state;
        }

        for (const long long state: keep) pool.del(state);
        for (int i = new_keep.size(); i < score_keep.size(); ++i) {
          sub_pool.del(score_keep[i]);
        }

        swap(keep, new_keep);
        const long long best_state = *min_element(keep.begin(), keep.end(), [&] (const long long &l, const long long &r) {
          return (*pool.get(l)) < (*pool.get(r));
        });
        if (verbose) {
          // pool.get(best_state)->print();
          cout << "Score = " << pool.get(best_state)->get_score() << endl << endl;
        }
        if (pool.get(best_state)->is_done()) break;
      }

      const long long best_state = *min_element(keep.begin(), keep.end(), [&] (const long long &l, const long long &r) {
        return (*pool.get(l)) < (*pool.get(r));
      });
      result.score = pool.get(best_state)->get_score();
      long long substate = pool.get(best_state)->substate_id;
      while (substate != -1) {
        result.history.emplace_back(sub_pool.get(substate)->action);
        substate = sub_pool.get(substate)->par;
      }
      reverse(result.history.begin(), result.history.end());
      return result;
    }

    static inline Result run(Param &param, const bool verbose=false) {
      /*
      - 一度のビームサーチで、何ターン先を決めるか
        - 幅、深さ
        - 何ターン先まで読むか
      */

      Result result;
      param.init();

      const long long best_state = pool.gen();
      pool.get(best_state)->init();
      pool.get(best_state)->substate_id = -1;

      for (int turn = 0; turn < param.get_max_turn(); turn += param.get_decide_turn()) {
        if (verbose) cout << "# turn : " << turn << endl;

        vector<long long> keep;
        const int beam_depth = param.get_beam_depth(turn);
        const int beam_width = param.get_beam_width(turn);
        __gnu_pbds::gp_hash_table<HashType, uint8_t> seen;

        { // init
          const long long init_state = pool.copy(best_state);
          seen[init_state] = 0;
          pool.get(init_state)->substate_id = -1;
          keep = {init_state};
        }

        for (int beam_turn = 0; beam_turn < beam_depth; ++beam_turn) {
          vector<long long> score_keep;
          calc_next_beam(keep, turn, seen, score_keep);
          const int w = min((int)score_keep.size(), beam_width);
          nth_element(score_keep.begin(), score_keep.begin() + w, score_keep.end(), [&] (const long long &l, const long long &r) {
            return sub_pool.get(l)->score < sub_pool.get(r)->score;
          });
          vector<long long> new_keep;

          for (int i = 0; i < w; ++i) {
            long long substate = score_keep[i];
            long long nowstate = sub_pool.get(substate)->state;
            long long new_state = pool.copy(nowstate);
            pool.get(new_state)->apply_op(sub_pool.get(substate)->action);
            pool.get(new_state)->substate_id = substate;
            sub_pool.get(substate)->par = pool.get(nowstate)->substate_id;
            new_keep.emplace_back(new_state);
          }

          for (const long long state: keep) pool.del(state);
          for (int i = w; i < score_keep.size(); ++i) {
            sub_pool.del(score_keep[i]);
          }

          swap(keep, new_keep);
          const long long this_best_state = *min_element(keep.begin(), keep.end(), [&] (const long long &l, const long long &r) {
            return (*pool.get(l)) < (*pool.get(r));
          });
          if (pool.get(this_best_state)->is_done()) break;
        }

        const long long this_best_state = *min_element(keep.begin(), keep.end(), [&] (const long long &l, const long long &r) {
          return (*pool.get(l)) < (*pool.get(r));
        });
        long long substate = pool.get(this_best_state)->substate_id;
        vector<Action> history;
        while (substate != -1) {
          history.emplace_back(sub_pool.get(substate)->action);
          substate = sub_pool.get(substate)->par;
        }
        sub_pool.clear();
        reverse(history.begin(), history.end());
        for (int i = 0; i < min(param.get_decide_turn(), (int)(history.size())); ++i) {
          result.history.emplace_back(history[i]);
          pool.get(best_state)->apply_op(history[i]);
          if (verbose) {
            pool.get(best_state)->print();
            cout << "Score = " << pool.get(best_state)->get_score() << endl << endl;
          }
        }
        if (pool.get(best_state)->is_done()) break;
      }
      result.score = pool.get(best_state)->get_score();
      return result;
    }
  };


  void advance(vector<vector<int>> &a, Result &result) {
    rep(i, N) rep(j, N) {
      if (i<GLOBAL_DO_UB || j<GLOBAL_DO_UB) {
        is_active[i*N+j] = 0;
      }
    }

    int zi, zj;
    rep(i, N) rep(j, N) {
      if (a[i][j] == 0) {
        zi = i;
        zj = j;
        break;
      }
    }
    for (Action action: result.history) {
      int ni = zi, nj = zj;
      switch (action) {
        case Action::D: ++ni; break;
        case Action::U: --ni; break;
        case Action::R: ++nj; break;
        case Action::L: --nj; break;
      }
      swap(a[zi][zj], a[ni][nj]);
      zi = ni;
      zj = nj;
    }
  }
}
}

void solve() {
  titan23::Timer timer;

  const int TL = -1;
  int TURN, DEP, W;

  vector<titan23::puzzle15::Result> ans;
  vector<int> UB = {N};
  if (N <= 4) {
    TURN = 80;
    DEP = TURN;
    W = 10'000;
  } else if (N <= 5) {
    TURN = 150;
    DEP = TURN;
    W = 10'000;
  } else if (N <= 10) {
    // 2sec 1108
    // TURN = 1000;
    // DEP = 300;
    // W = 285;
    // UB = {1,3,5,N};

    // 40sec 1006
    TURN = 2000;
    DEP = 2000;
    W = 500;
    UB = {N};
  } else {
    cerr << N << endl;
    assert(false);
  }

  for (int ub: UB) {
    GLOBAL_DO_UB = ub;
    titan23::puzzle15::init();
    titan23::puzzle15::Param param(TL, TURN, 50, DEP, W, false);
    titan23::puzzle15::Result res = titan23::puzzle15::BeamSearch::run(param, LOGS);
    // titan23::puzzle15::Result res = titan23::puzzle15::run_each_turn(param, LOGS);
    ans.emplace_back(res);
    titan23::puzzle15::advance(A, res);
  }

  int ans_size = 0;
  for (auto &res: ans) {
    ans_size += res.history.size();
  }
  cout << ans_size << endl;
  for (auto &res: ans) {
    res.print("");
  }
  cout << endl;

  cerr << timer.elapsed() << " ms" << endl;
}

int main() {
  input();
  solve();
  return 0;
}
