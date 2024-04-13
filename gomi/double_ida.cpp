// %%file main.cpp
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")

#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <string>
#include <stack>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include "./random.cpp"
#include "./action.cpp"
#include "./util.cpp"
#include "./timer.cpp"
using namespace std;

#define rep(i, n) for (int i = 0; i < (n); ++i)
using HashType = unsigned long long;

// search_dep=55 8.56286

const int INF = 2e9;

int N;
vector<vector<int>> A;
vector<pair<int, int>> revA;
vector<int> revA_i, revA_j;
vector<vector<vector<HashType>>> hash_rand;

void input() {
  cin >> N;
  A.resize(N, vector<int>(N));
  revA.resize(N*N);
  revA_i.resize(N*N);
  revA_j.resize(N*N);
  rep(i, N) rep(j, N) {
    cin >> A[i][j];
    revA[A[i][j]] = {i, j};
    revA_i[A[i][j]] = i;
    revA_j[A[i][j]] = j;
  }
  titan23::Random r;
  hash_rand.resize(N, vector<vector<HashType>>(N, vector<HashType>(N*N)));
  rep(i, N) rep(j, N) {
    rep(num, N*N) {
      hash_rand[i][j][num] = r.randrand();
    }
  }
}

namespace titan23 {

namespace IDA {
  unordered_map<HashType, int> wd_ud_start, wd_lr_start;
  unordered_map<HashType, int> wd_ud_end, wd_lr_end;
  vector<vector<vector<HashType>>> wd_hash_rand;
  vector<HashType> pos_hash_rand;

  struct State {
    vector<vector<short>> wd;
    int pos;
    HashType hash;
    State() : wd(N, vector<short>(N, 0)) {}

    void init() {
      wd.resize(N, vector<short>(N, 0));
      titan23::Random r;
      wd_hash_rand.resize(N, vector<vector<HashType>>(N, vector<HashType>(N+1)));
      rep(i, N) rep(j, N) rep(k, N+1) {
        wd_hash_rand[i][j][k] = r.randrand();
      }
      pos_hash_rand.resize(N);
      rep(i, N) {
        pos_hash_rand[i] = r.randrand();
      }
    }

    void print() const {
      rep(i, N) {
        rep(j, N) {
          cout << wd[i][j] << ' ';
        }
        cout << endl;
      }
      cout << endl;
    }

    void calc_hash() {
      hash = pos_hash_rand[pos];
      rep(i, N) rep(j, N) {
        hash ^= wd_hash_rand[i][j][wd[i][j]];
      }
    }

    HashType try_up(const int j) const {
      HashType h = hash;
      h ^= wd_hash_rand[pos][j][wd[pos][j]];
      h ^= wd_hash_rand[pos-1][j][wd[pos-1][j]];
      h ^= wd_hash_rand[pos][j][wd[pos][j]+1];
      h ^= wd_hash_rand[pos-1][j][wd[pos-1][j]-1];
      h ^= pos_hash_rand[pos];
      h ^= pos_hash_rand[pos-1];
      return h;
    }

    HashType try_down(const int j) const {
      HashType h = hash;
      h ^= wd_hash_rand[pos][j][wd[pos][j]];
      h ^= wd_hash_rand[pos+1][j][wd[pos+1][j]];
      h ^= wd_hash_rand[pos][j][wd[pos][j]+1];
      h ^= wd_hash_rand[pos+1][j][wd[pos+1][j]-1];
      h ^= pos_hash_rand[pos];
      h ^= pos_hash_rand[pos+1];
      return h;
    }

    void apply_up(const int i) {
      hash = try_up(i);
      ++wd[pos][i];
      --wd[pos-1][i];
      --pos;
    }

    void apply_down(const int i) {
      hash = try_down(i);
      ++wd[pos][i];
      --wd[pos+1][i];
      ++pos;
    }

    void apply_left(const int i) {
      apply_up(i);
    }

    void apply_right(const int i) {
      apply_down(i);
    }

    HashType tohash() const {
      return hash;
    }
  };

  void calc_WD() {
    State state;
    state.init();

    rep(i, N) state.wd[i][i] = N;
    state.wd[N-1][N-1] = N-1;
    state.pos = N-1;
    state.calc_hash();

    wd_ud_start[state.tohash()] = 0;
    wd_lr_start[state.tohash()] = 0;
    queue<pair<int, State>> qu;
    qu.emplace(0, state);
    while (!qu.empty() && wd_ud_start.size() < 1e5) {
      int dist = qu.front().first;
      state = qu.front().second;
      qu.pop();
      if (state.pos != 0) { // up
        rep(j, N) {
          if (state.wd[state.pos-1][j]) {
            HashType h = state.try_up(j);
            if (wd_ud_start.find(h) != wd_ud_start.end()) continue;
            State new_state = state;
            new_state.apply_up(j);
            wd_ud_start[new_state.tohash()] = dist+1;
            wd_lr_start[new_state.tohash()] = dist+1;
            qu.emplace(dist+1, new_state);
          }
        }
      }

      if (state.pos != N-1) { // down
        rep(j, N) {
          if (state.wd[state.pos+1][j]) {
            HashType h = state.try_down(j);
            if (wd_ud_start.find(h) != wd_ud_start.end()) continue;
            State new_state = state;
            new_state.apply_down(j);
            wd_ud_start[new_state.tohash()] = dist+1;
            wd_lr_start[new_state.tohash()] = dist+1;
            qu.emplace(dist+1, new_state);
          }
        }
      }
    }

    rep(i, N) rep(j, N) {
    }

    cerr << wd_ud_start.size() << ' ' << wd_lr_start.size() << endl;
    // exit(0);
  }

  struct Node {
    State state_memo_start_ud, state_memo_start_lr;
    HashType hash;
    vector<vector<int>> field;
    vector<Action> history;
    int score_md_start, score_md_end, score_true;
    int wd_dist, inv_dist;
    int zero_i, zero_j;
    string type;
    Node() {}

    int get_pos_score_start(int i, int j, int val) const {
      if (val == 0) return 0;
      int vi = (val-1) / N;
      int vj = (val-1) % N;
      return abs(i - vi) + abs(j - vj);
    }

    int get_pos_score_end(int i, int j, int val) const {
      if (val == 0) return 0;
      auto [vi, vj] = revA[val];
      return abs(i - vi) + abs(j - vj);
    }

    pair<int, int> calc_predcost_start() {
      int score = 0;
      rep(i, N) rep(j, N) {
        score += get_pos_score_start(i, j, field[i][j]);
      }
      int inv_dist = 0;
      wd_dist = 0;
      {
        rep(i, N) rep(j, N) {
          state_memo_start_ud.wd[i][j] = 0;
        }
        state_memo_start_ud.pos = zero_i;
        rep(i, N) rep(j, N) {
          int val = field[i][j];
          if (val == 0) continue;
          int vi = (val-1) / N;
          state_memo_start_ud.wd[i][vi]++;
        }
        state_memo_start_ud.calc_hash();
        HashType h = state_memo_start_ud.tohash();
        if (wd_ud_start.find(h) != wd_ud_start.end()) wd_dist += wd_ud_start[h];
      }
      {
        rep(i, N) rep(j, N) {
          state_memo_start_lr.wd[i][j] = 0;
        }
        state_memo_start_lr.pos = zero_j;
        rep(i, N) rep(j, N) {
          int val = field[i][j];
          if (val == 0) continue;
          int vj = revA[val].second;
          state_memo_start_lr.wd[j][vj]++;
        }
        state_memo_start_lr.calc_hash();
        HashType h = state_memo_start_lr.tohash();
        if (wd_lr_start.find(h) != wd_lr_start.end()) wd_dist += wd_lr_start[h];
      }
      return {score, wd_dist};
    }

    int calc_predcost_end() const {
      int score = 0;
      rep(i, N) rep(j, N) {
        score += get_pos_score_end(i, j, field[i][j]);
      }
      return score;
    }

    int get_predcost_end() {
      // return score_md_end;  // for md
      return score_md_end;  // for md
    }

    int get_predcost_start() {
      // return score_md_start;
      // int inv = get_inversion_cnt();
      int inv_dist = 0;
      // inv_dist = inv/3 + inv%3;
      // auto [ms, ws] = calc_predcost_start();
      // assert(score_md_start == ms);
      // assert(wd_dist == ws);
      return max(max(wd_dist, inv_dist), score_md_start);
    }

    int get_cost() const {
      return score_true;
    }

    void move(const Action &action) {
      switch (action) {
      case Action::D:
        if (type == "start") {
          {
            auto it = wd_ud_start.find(state_memo_start_ud.tohash());
            if (it != wd_ud_start.end()) wd_dist -= it->second;
          }
          state_memo_start_ud.apply_down((field[zero_i+1][zero_j]-1)/N);
          {
            auto it = wd_ud_start.find(state_memo_start_ud.tohash());
            if (it != wd_ud_start.end()) wd_dist += it->second;
          }
          score_md_start -= get_pos_score_start(zero_i, zero_j, field[zero_i][zero_j]);
          score_md_start += get_pos_score_start(zero_i, zero_j, field[zero_i+1][zero_j]);
          score_md_start -= get_pos_score_start(zero_i+1, zero_j, field[zero_i+1][zero_j]);
          score_md_start += get_pos_score_start(zero_i+1, zero_j, field[zero_i][zero_j]);
        }
        if (type == "end") {
          score_md_end -= get_pos_score_end(zero_i, zero_j, field[zero_i][zero_j]);
          score_md_end += get_pos_score_end(zero_i, zero_j, field[zero_i+1][zero_j]);
          score_md_end -= get_pos_score_end(zero_i+1, zero_j, field[zero_i+1][zero_j]);
          score_md_end += get_pos_score_end(zero_i+1, zero_j, field[zero_i][zero_j]);
        }
        hash ^= hash_rand[zero_i][zero_j][field[zero_i][zero_j]];
        hash ^= hash_rand[zero_i][zero_j][field[zero_i+1][zero_j]];
        hash ^= hash_rand[zero_i+1][zero_j][field[zero_i+1][zero_j]];
        hash ^= hash_rand[zero_i+1][zero_j][field[zero_i][zero_j]];
        swap(field[zero_i][zero_j], field[zero_i+1][zero_j]);
        ++zero_i; break;
      case Action::R:
        if (type == "start") {
          {
            auto it = wd_lr_start.find(state_memo_start_lr.tohash());
            if (it != wd_lr_start.end()) wd_dist -= it->second;
          }
          state_memo_start_lr.apply_right((field[zero_i][zero_j+1]-1)%N);
          {
            auto it = wd_lr_start.find(state_memo_start_lr.tohash());
            if (it != wd_lr_start.end()) wd_dist += it->second;
          }
          score_md_start -= get_pos_score_start(zero_i, zero_j, field[zero_i][zero_j]);
          score_md_start += get_pos_score_start(zero_i, zero_j, field[zero_i][zero_j+1]);
          score_md_start -= get_pos_score_start(zero_i, zero_j+1, field[zero_i][zero_j+1]);
          score_md_start += get_pos_score_start(zero_i, zero_j+1, field[zero_i][zero_j]);
        }
        if (type == "end") {
          score_md_end -= get_pos_score_end(zero_i, zero_j, field[zero_i][zero_j]);
          score_md_end += get_pos_score_end(zero_i, zero_j, field[zero_i][zero_j+1]);
          score_md_end -= get_pos_score_end(zero_i, zero_j+1, field[zero_i][zero_j+1]);
          score_md_end += get_pos_score_end(zero_i, zero_j+1, field[zero_i][zero_j]);
        }
        hash ^= hash_rand[zero_i][zero_j][field[zero_i][zero_j]];
        hash ^= hash_rand[zero_i][zero_j][field[zero_i][zero_j+1]];
        hash ^= hash_rand[zero_i][zero_j+1][field[zero_i][zero_j+1]];
        hash ^= hash_rand[zero_i][zero_j+1][field[zero_i][zero_j]];
        swap(field[zero_i][zero_j], field[zero_i][zero_j+1]);
        ++zero_j; break;
      case Action::U:
        if (type == "start") {
          {
            auto it = wd_ud_start.find(state_memo_start_ud.tohash());
            if (it != wd_ud_start.end()) wd_dist -= it->second;
          }
          state_memo_start_ud.apply_up((field[zero_i-1][zero_j]-1)/N);
          {
            auto it = wd_ud_start.find(state_memo_start_ud.tohash());
            if (it != wd_ud_start.end()) wd_dist += it->second;
          }
          score_md_start -= get_pos_score_start(zero_i, zero_j, field[zero_i][zero_j]);
          score_md_start += get_pos_score_start(zero_i, zero_j, field[zero_i-1][zero_j]);
          score_md_start -= get_pos_score_start(zero_i-1, zero_j, field[zero_i-1][zero_j]);
          score_md_start += get_pos_score_start(zero_i-1, zero_j, field[zero_i][zero_j]);
        }
        if (type == "end") {
          score_md_end -= get_pos_score_end(zero_i, zero_j, field[zero_i][zero_j]);
          score_md_end += get_pos_score_end(zero_i, zero_j, field[zero_i-1][zero_j]);
          score_md_end -= get_pos_score_end(zero_i-1, zero_j, field[zero_i-1][zero_j]);
          score_md_end += get_pos_score_end(zero_i-1, zero_j, field[zero_i][zero_j]);
        }
        hash ^= hash_rand[zero_i][zero_j][field[zero_i][zero_j]];
        hash ^= hash_rand[zero_i][zero_j][field[zero_i-1][zero_j]];
        hash ^= hash_rand[zero_i-1][zero_j][field[zero_i-1][zero_j]];
        hash ^= hash_rand[zero_i-1][zero_j][field[zero_i][zero_j]];
        swap(field[zero_i][zero_j], field[zero_i-1][zero_j]);
        --zero_i; break;
      case Action::L:
        if (type == "start") {
          {
            auto it = wd_lr_start.find(state_memo_start_lr.tohash());
            if (it != wd_lr_start.end()) wd_dist -= it->second;
          }
          state_memo_start_lr.apply_left((field[zero_i][zero_j-1]-1)%N);
          {
            auto it = wd_lr_start.find(state_memo_start_lr.tohash());
            if (it != wd_lr_start.end()) wd_dist += it->second;
          }
          score_md_start -= get_pos_score_start(zero_i, zero_j, field[zero_i][zero_j]);
          score_md_start += get_pos_score_start(zero_i, zero_j, field[zero_i][zero_j-1]);
          score_md_start -= get_pos_score_start(zero_i, zero_j-1, field[zero_i][zero_j-1]);
          score_md_start += get_pos_score_start(zero_i, zero_j-1, field[zero_i][zero_j]);
        }
        if (type == "end") {
          score_md_end -= get_pos_score_end(zero_i, zero_j, field[zero_i][zero_j]);
          score_md_end += get_pos_score_end(zero_i, zero_j, field[zero_i][zero_j-1]);
          score_md_end -= get_pos_score_end(zero_i, zero_j-1, field[zero_i][zero_j-1]);
          score_md_end += get_pos_score_end(zero_i, zero_j-1, field[zero_i][zero_j]);
        }
        hash ^= hash_rand[zero_i][zero_j][field[zero_i][zero_j]];
        hash ^= hash_rand[zero_i][zero_j][field[zero_i][zero_j-1]];
        hash ^= hash_rand[zero_i][zero_j-1][field[zero_i][zero_j-1]];
        hash ^= hash_rand[zero_i][zero_j-1][field[zero_i][zero_j]];
        swap(field[zero_i][zero_j], field[zero_i][zero_j-1]);
        --zero_j; break;
      default: assert(false);
      }
    }

    void rollback(const Action &action) {
      --score_true;
      history.pop_back();
      move(get_rev_action(action));
    }

    void apply_op(const Action &action) {
      ++score_true;
      history.emplace_back(action);
      move(action);
    }

    vector<Action> get_actions() {
      vector<Action> res;
      if (zero_i-1 >= 0) res.emplace_back(Action::U);
      if (zero_i+1 < N)  res.emplace_back(Action::D);
      if (zero_j-1 >= 0) res.emplace_back(Action::L);
      if (zero_j+1 < N)  res.emplace_back(Action::R);
      if (!history.empty()) {
        Action rev = get_rev_action(history.back());
        res.erase(remove(res.begin(), res.end(), rev), res.end());
      }
      return res;
    }

    void print() {
      rep(i, N) {
        rep(j, N) {
          cout << zfill(field[i][j], 2) << ' ';
        }
        cout << endl;
      }
      cout << endl;
    }

    int get_inversion_cnt() const {
      int cnt = 0;
      rep(i, N*N-1) {
        for (int j = i+1; j < N*N-1; ++j) {
          if (field[i/N][i%N] > field[j/N][j%N]) {
            ++cnt;
          }
        }
      }
      return cnt;
    }

    void init_start() {
      type = "start";
      history.clear();
      field = A;
      hash = 0;
      score_true = 0;
      rep(i, N) rep(j, N) {
        field[i][j] = A[i][j];
        hash ^= hash_rand[i][j][field[i][j]];
        if (field[i][j] == 0) {
          zero_i = i;
          zero_j = j;
        }
      }
      tie(score_md_start, wd_dist) = calc_predcost_start();
    }

    void init_end() {
      type = "end";
      history.clear();
      field.resize(N, vector<int>(N));
      hash = 0;
      score_true = 0;
      rep(i, N) rep(j, N) {
        field[i][j] = i*N+j + 1;
        if (i == N-1 && j == N-1) {
          zero_i = N-1;
          zero_j = N-1;
          field[i][j] = 0;
        }
        hash ^= hash_rand[i][j][field[i][j]];
      }
      score_md_end = calc_predcost_end();
    }
  };

  Node node_start;
  Node node_end;
  vector<Action> best_history;
  int dfs_limit_start;
  int dfs_limit_end;
  int dfs_limit;
  int tmp_best_cost;
  HashType find_hash;
  unordered_map<HashType, int> seen_from_start, seen_dfs_end;

  bool dfs_start() {
    if (node_start.get_cost() + node_start.get_predcost_start() > dfs_limit_start) return false;
    if (seen_dfs_end.find(node_start.hash) != seen_dfs_end.end()) {
      int d = node_start.history.size() + seen_dfs_end[node_start.hash];
      if (d == dfs_limit_start) {
        best_history = node_start.history;
        find_hash = node_start.hash;
        return true;
      } else if (d < tmp_best_cost) {
        tmp_best_cost = d;
        cout << "tmp_best_cost=" << tmp_best_cost << endl;
      }
    }
    if (seen_from_start.find(node_start.hash) != seen_from_start.end() && seen_from_start[node_start.hash] <= node_start.get_cost()) return false;
    seen_from_start[node_start.hash] = node_start.get_cost();
    vector<Action> actions = node_start.get_actions();
    for (const Action &action: actions) {
      node_start.apply_op(action);
      if (dfs_start()) return true;
      node_start.rollback(action);
    }
    return false;
  }

  void dfs_end() {
    if (node_end.get_cost() + node_end.get_predcost_end() > dfs_limit_end) return;
    if (seen_dfs_end.find(node_end.hash) != seen_dfs_end.end() && seen_dfs_end[node_end.hash] <= node_end.get_cost()) return;
    seen_dfs_end[node_end.hash] = node_end.get_cost();
    vector<Action> actions = node_end.get_actions();
    for (const Action &action: actions) {
      node_end.apply_op(action);
      dfs_end();
      node_end.rollback(action);
    }
  }

  pair<bool, vector<Action>> dfs_end_hashsearch(HashType hash) {
    if (node_end.get_cost() + node_end.get_predcost_end() > dfs_limit_end) return {false, {}};
    if (seen_dfs_end.find(node_end.hash) != seen_dfs_end.end() && seen_dfs_end[node_end.hash] <= node_end.get_cost()) return {false, {}};
    if (node_end.hash == hash) {
      vector<Action> res = node_end.history;
      return {true, res};
    }
    seen_dfs_end[node_end.hash] = node_end.get_cost();
    vector<Action> actions = node_end.get_actions();
    for (const Action &action: actions) {
      node_end.apply_op(action);
      pair<bool, vector<Action>> result = dfs_end_hashsearch(hash);
      if (result.first) {
        return result;
      }
      node_end.rollback(action);
    }
    return {false, {}};
  }

  vector<Action> run() {
    calc_WD();
    node_start.state_memo_start_ud.wd.resize(N, vector<short>(N, 0));
    node_start.state_memo_start_lr.wd.resize(N, vector<short>(N, 0));
    node_end.state_memo_start_ud.wd.resize(N, vector<short>(N, 0));
    node_end.state_memo_start_lr.wd.resize(N, vector<short>(N, 0));

    tmp_best_cost = INF;
    node_start.init_start();
    assert(node_start.state_memo_start_ud.pos == node_start.zero_i);
    if (node_start.get_predcost_start() == 0) return {};
    dfs_limit_start = node_start.get_predcost_start()/2;
    int inv = node_start.get_inversion_cnt();
    if (dfs_limit_start%2 != inv%2) ++dfs_limit_start;
    dfs_limit_end = dfs_limit_start;
    seen_dfs_end.clear();
    node_end.init_end();
    seen_dfs_end[node_end.hash] = 0;
    dfs_end();

    titan23::Timer timer;
    while (true) {
      dfs_limit = max(dfs_limit_start, dfs_limit_end);
      cerr << "search_dep=" << dfs_limit_start << endl;

      node_start.init_start();
      seen_from_start.clear();
      if (dfs_start()) {
        // node_end.init_end();
        // seen_dfs_end.clear();
        // dfs_end_hashsearch(find_hash);
        // vector<Action> end_hist = dfs_end_hashsearch(find_hash).second;
        // reverse(end_hist.begin(), end_hist.end());
        // for (Action action: end_hist) {
        //   best_history.emplace_back(get_rev_action(action));
        // }
        return best_history;
      }
      dfs_limit_start += 2;
      // if (seen_dfs_end.size() < 1e6) {
      //   dfs_limit_end += 2;
      //   node_end.init_end();
      //   seen_dfs_end.clear();
      //   dfs_end();
      // }
      cerr << timer.elapsed()/1000 << endl;
    }
    assert(false);
  }
}
}

void solve() {
  titan23::Timer timer;
  vector<titan23::Action> ans = titan23::IDA::run();
  cout << ans.size() << endl;
  for (const titan23::Action &action: ans) {
    cout << action;
  }
  cout << endl;
  cout << timer.elapsed() << "msec." << endl;
}

int main() {
  input();
  solve();
  return 0;
}
