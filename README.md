# IP演習
production > main.pyが最終的な完成形

基本的にはプログラムを実行したら画面に操作の指示が出るのでそれに従えばOKです．不明点などあったら気軽に質問してください

## 実行環境の作成
git cloneでプロジェクトをローカル環境にダウンロードしてください．

pyproject.tomlというファイルに必要なライブラリとそのバージョンが記載されています．

pipコマンドまたはpoetryを使ってインストールしてください．

### パターン1: pipコマンドを使ったインストール方法
```
pip install torch==1.13 openpifpaf==0.13.8 opencv-python==4.6.0.66
```

### パターン2: poetryを使ったインストール方法
- poetryの導入
```
brew install poetry
```

- pyproject.tomlがある階層で以下のコマンド
```
poetry install
```

- 以降，以下のコマンドだけでライブラリが用意された環境に入れるようになる
```
poetry shell
```