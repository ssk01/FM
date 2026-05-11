Training Logs — MF / FeatMF / MF+SideFeatures
==============================================

Common setup:
  Dataset: MovieLens 100k (943 users, 1682 items, 100k ratings)
  Binary label: rating >= 4 → 1, else 0
  Split: 80/10/10
  Optimizer: Adam(lr=0.01, weight_decay=1e-5), 20 epochs
  Embedding dim: 64

--------------------------------------------------------------------------------
Model 1: Pure MF (mf.py)
  Features: user_id, item_id
  Params: 170,626
  Test AUC: 0.7364
--------------------------------------------------------------------------------
  E 1  train_loss=0.6377 auc=0.7184  val_loss=0.5719 auc=0.7797  ★
  E 2  train_loss=0.5095 auc=0.8335  val_loss=0.5419 auc=0.7951  ★
  E 3  train_loss=0.3917 auc=0.9201  val_loss=0.5571 auc=0.7844
  E 4  train_loss=0.2864 auc=0.9697  val_loss=0.5903 auc=0.7692
  E 5  train_loss=0.2089 auc=0.9895  val_loss=0.6277 auc=0.7583
  E 6  train_loss=0.1507 auc=0.9979  val_loss=0.6437 auc=0.7553
  E 7  train_loss=0.1328 auc=0.9990  val_loss=0.6594 auc=0.7524
  E 8  train_loss=0.1185 auc=0.9995  val_loss=0.6742 auc=0.7498
  E 9  train_loss=0.1070 auc=0.9998  val_loss=0.6879 auc=0.7477
  E10  train_loss=0.0976 auc=0.9999  val_loss=0.7002 auc=0.7459
  E11  train_loss=0.0866 auc=1.0000  val_loss=0.7054 auc=0.7453
  E12  train_loss=0.0837 auc=1.0000  val_loss=0.7104 auc=0.7446
  E13  train_loss=0.0812 auc=1.0000  val_loss=0.7152 auc=0.7440
  E14  train_loss=0.0786 auc=1.0000  val_loss=0.7201 auc=0.7432
  E15  train_loss=0.0764 auc=1.0000  val_loss=0.7242 auc=0.7426
  E16  train_loss=0.0727 auc=1.0000  val_loss=0.7262 auc=0.7424
  E17  train_loss=0.0718 auc=1.0000  val_loss=0.7279 auc=0.7422
  E18  train_loss=0.0709 auc=1.0000  val_loss=0.7299 auc=0.7419
  E19  train_loss=0.0701 auc=1.0000  val_loss=0.7316 auc=0.7417
  E20  train_loss=0.0693 auc=1.0000  val_loss=0.7333 auc=0.7415

  → 过拟合严重，E2 达到峰值后一路下降。纯 ID embedding 容量太大。

--------------------------------------------------------------------------------
Model 2: FeatMF — No user ID (mf_feature.py)
  Features: occupation one-hot → projection, item_id, genre
  Params: 111,913
  Test AUC: 0.6774
--------------------------------------------------------------------------------
  E 1  train_loss=0.6346 auc=0.6822  val_loss=0.6179 auc=0.7113  ★
  E 2  train_loss=0.6042 auc=0.7277  val_loss=0.6171 auc=0.7109
  E 3  train_loss=0.5866 auc=0.7477  val_loss=0.6284 auc=0.7008
  E 4  train_loss=0.5694 auc=0.7651  val_loss=0.6402 auc=0.6934
  E 5  train_loss=0.5563 auc=0.7769  val_loss=0.6538 auc=0.6849
  E 6  train_loss=0.5485 auc=0.7835  val_loss=0.6640 auc=0.6802
  E 7  train_loss=0.5423 auc=0.7879  val_loss=0.6684 auc=0.6791
  E 8  train_loss=0.5397 auc=0.7904  val_loss=0.6766 auc=0.6754
  E 9  train_loss=0.5368 auc=0.7911  val_loss=0.6778 auc=0.6760
  E10  train_loss=0.5343 auc=0.7929  val_loss=0.6820 auc=0.6734
  E11  train_loss=0.5342 auc=0.7934  val_loss=0.6828 auc=0.6740
  E12  train_loss=0.5337 auc=0.7940  val_loss=0.6852 auc=0.6737
  E13  train_loss=0.5332 auc=0.7941  val_loss=0.6871 auc=0.6715
  E14  train_loss=0.5312 auc=0.7947  val_loss=0.6856 auc=0.6746
  E15  train_loss=0.5309 auc=0.7954  val_loss=0.6872 auc=0.6758
  E16  train_loss=0.5307 auc=0.7964  val_loss=0.6871 auc=0.6737
  E17  train_loss=0.5300 auc=0.7960  val_loss=0.6889 auc=0.6724
  E18  train_loss=0.5294 auc=0.7967  val_loss=0.6913 auc=0.6710
  E19  train_loss=0.5303 auc=0.7963  val_loss=0.6915 auc=0.6727
  E20  train_loss=0.5293 auc=0.7971  val_loss=0.6893 auc=0.6754

  → 无 user_id，仅靠 occupation（21 种）区分用户，AUC 最高不到 0.68，远不如纯 MF。
    没有过拟合迹象（训练集 AUC 不到 0.80），模型 capacity 不足。

--------------------------------------------------------------------------------
Model 3: MF + Side Features (fm_demo.py)
  Features: user_id, occupation, item_id, genre
  Params: ~280K
  Test AUC: 0.7929
--------------------------------------------------------------------------------
  E 1  train_loss=0.5988 auc=0.7333  val_loss=0.5527 auc=0.7858  ★
  E 2  train_loss=0.5197 auc=0.8152  val_loss=0.5437 auc=0.7934  ★
  E 3  train_loss=0.4599 auc=0.8626  val_loss=0.5544 auc=0.7903
  E 4  train_loss=0.3947 auc=0.9050  val_loss=0.5784 auc=0.7833
  E 5  train_loss=0.3288 auc=0.9400  val_loss=0.6140 auc=0.7750
  E 6  train_loss=0.2540 auc=0.9718  val_loss=0.6326 auc=0.7728
  E 7  train_loss=0.2251 auc=0.9805  val_loss=0.6542 auc=0.7700
  E 8  train_loss=0.2011 auc=0.9863  val_loss=0.6750 auc=0.7667
  E 9  train_loss=0.1807 auc=0.9906  val_loss=0.6943 auc=0.7648
  E10  train_loss=0.1616 auc=0.9938  val_loss=0.7160 auc=0.7629
  E11  train_loss=0.1389 auc=0.9970  val_loss=0.7243 auc=0.7629
  E12  train_loss=0.1317 auc=0.9977  val_loss=0.7350 auc=0.7619
  E13  train_loss=0.1253 auc=0.9982  val_loss=0.7446 auc=0.7617
  E14  train_loss=0.1189 auc=0.9987  val_loss=0.7543 auc=0.7610
  E15  train_loss=0.1132 auc=0.9990  val_loss=0.7640 auc=0.7606
  E16  train_loss=0.1051 auc=0.9994  val_loss=0.7687 auc=0.7600
  E17  train_loss=0.1029 auc=0.9995  val_loss=0.7739 auc=0.7597
  E18  train_loss=0.1004 auc=0.9996  val_loss=0.7783 auc=0.7595
  E19  train_loss=0.0979 auc=0.9996  val_loss=0.7834 auc=0.7591
  E20  train_loss=0.0959 auc=0.9997  val_loss=0.7883 auc=0.7590

  → 最佳模型。加了 occupation + genre side features 后，过拟合被明显缓解，
    peak AUC 0.793 且下降更平缓。test AUC 0.793。

--------------------------------------------------------------------------------
Summary
--------------------------------------------------------------------------------
  Model                          Test AUC   Params     Best Val Epoch
  Pure MF (user_id only)          0.736     170,626    2
  FeatMF (occupation only)        0.677     111,913    1-2
  MF + Side Features              0.793     280,000    2

  Side features 贡献约 +0.06 AUC（对比纯 MF），
  user_id 贡献约 +0.06 AUC（对比 FeatMF）。
  两者叠加效果最好。
