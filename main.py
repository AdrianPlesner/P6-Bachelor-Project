import h5_to_gluonts as hg
train, test = hg.load_h5_to_gluon("./data/pems-bay.h5", train_size=288*31, test_size=288, freq="5Min")
hg.plot_train_test(train, test)
