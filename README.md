# MNIST 手書き数字認識 Webアプリ

このリポジトリは、FastAPI・TensorFlow・ngrokを用いた手書き数字認識Webアプリです。  
ユーザーが描いた数字画像をCNNで認識し、中間層の活性化画像も可視化します。

## 主な機能

- 手書き数字画像の認識（MNISTデータセットで学習済みCNNモデル）
- 中間層の活性化画像をWeb上で表示
- ngrokによる一時的な公開URL生成
- FastAPIによるAPI・Webサーバー
- データ拡張によるモデル再学習（必要時のみ）

## ngrokの登録・トークン取得方法

このプロジェクトを利用するには、ngrokの認証トークンが必要です。  
まだngrokアカウントを作成していない方は、以下の手順で登録・トークン取得を行ってください。

1. [ngrok公式サイト](https://ngrok.com/) にアクセスし、アカウントを作成します。
2. ログイン後、ダッシュボードの「Your Authtoken」欄から認証トークンをコピーできます。
3. このトークンは、ngrokを使った公開URL生成の際に必要となります。

## ローカルでの実行方法

1. **依存パッケージのインストール**

   ```sh
   poetry install
   ```

2. **環境変数の設定**

   `.env` ファイルを作成し、以下のように設定してください（プロジェクトルートに配置）

   ```
   MODEL_PATH=models/mnist_cnn.h5
   NGROK_TOKEN=あなたのngrokトークン
   ```

   - `MODEL_PATH` : モデル保存先（デフォルト: `models/mnist_cnn.h5`）
   - `NGROK_TOKEN` : ngrokの認証トークン

3. **サーバー起動**

   ```sh
   poetry run python main.py
   ```

   起動後、ngrokの公開URLが表示されます。

## Google Colabでの実行方法

1. **リポジトリのダウンロード**

   ```python
   !git clone https://github.com/Ty-Yuki/handwritten-digit-recognition.git
   %cd handwritten-digit-recognition
   ```

2. **必要パッケージのインストール**

   ```python
   !pip install pyngrok
   ```

3. **ngrokトークンの設定**

   ```python
   import os
   from pyngrok import ngrok

   os.environ["NGROK_TOKEN"] = "あなたのngrokトークン"
   ngrok.set_auth_token(os.environ["NGROK_TOKEN"])
   ```

4. **サーバーの起動**

   ```python
   !python main.py
   ```

## 使い方

- ブラウザで公開URLにアクセス
- キャンバスに数字を描画し、「予測」ボタンを押すと、認識結果とCNN中間層の活性化画像が表示されます

## ディレクトリ構成

- `main.py` : サーバー起動・API定義
- `app/model_utils.py` : モデル構築・学習・保存
- `app/visualization.py` : 中間層活性化画像生成
- `templates/index.html` : Webフロントエンド
- `static/` : 静的ファイル（ロゴ等）

## 開発・テスト

- コード整形: `make fmt`
- Lint: `make lint`
- テスト: `make test`

## ライセンス

MIT
