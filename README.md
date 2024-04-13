# SlidingPuzzle

スライドパズルのソルバ

[ビジュアライザ](https://aktardigrade13.github.io/HUIT_SHINKAN2024/)

- [IDA](./sol/ida.cpp)
  - IDA* 探索による厳密解法
  - `N <= 4` を推奨
- [ビムサ](./sol/beam_search.cpp)
  - ビームサーチによるヒューリスティック解法
  - `N = 10` でも意外と動きそう、パラメータ調整必要
