# ML-Template(Pytorch Version)
このプロジェクトは、画像処理のディープラーニングモデルをPyTorchで構築するときのテンプレートプログラムです。プログラムはこのままでは適切に動作しない可能性があるので、ご自身の用意したデータセットに応じて適切にプログラムを修正し、ご利用ください。
## 特徴説明
- データセット
trainプログラム内ではCIFAR-10データセットを採用しています。ご自身のデータセットでモデルを構築するときはdatasets.datasetモジュールの中をご自身の環境に合わせて修正し、ご利用ください。
- ネットワーク
本プロジェクトでは、ResNetモデルを利用し、学習するためのテンプレートを提供しています。別のネットワークでモデルの構築を行いたいときはmodelsディレクトリの中を編集し、trainプログラムで適切に呼び出すようにしてください。
- 他

    本プロジェクトでは、以下の機能を実装したテンプレートを提供しています。
    
        - 早期終了
        - トレーニングとバリデーションの損失をプロットする機能
        - seedの固定機能
        - プロセス名の決定
        - パーサー
        - GPUの並列化
## Author
### [tech-tatsuma](https://github.com/tech-tatsuma)
