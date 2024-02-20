import time
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from model_new import MyNetwork, train, test, train_last, val
from torch.utils.data import ConcatDataset
from dataset import MyDataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

np.random.seed(config.seed)
random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.deterministic = True


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=config.epoch, help="epoch")
    parser.add_argument("--epoch_step", type=int, default=config.epoch_step, help="epoch_step")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="batch_size")
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate, default=0.01')
    parser.add_argument('--seed', type=int, default=config.seed, help='seed')
    parser.add_argument("--main_path", type=str, default=config.main_path, help="main_path")
    parser.add_argument("--dataset", type=str, default=config.dataset, help="dataset")
    parser.add_argument("--dataset_save_as", type=str, default=config.dataset, help="dataset_save_as")
    parser.add_argument("--max_natoms", type=int, default=config.max_natoms, help="max_natoms")
    parser.add_argument("--length", type=int, default=config.length, help="important: data length")
    parser.add_argument("--root_bmat", type=str, default=config.root_bmat, help="root_bmat")
    parser.add_argument("--root_dmat", type=str, default=config.root_dmat, help="root_dmat")
    parser.add_argument("--root_conf", type=str, default=config.root_conf, help="root_conf")
    parser.add_argument("--format_bmat", type=str, default=config.format_bmat, help="format_bmat")
    parser.add_argument("--format_dmat", type=str, default=config.format_dmat, help="format_dmat")
    parser.add_argument("--format_conf", type=str, default=config.format_conf, help="format_conf")
    parser.add_argument("--format_eigen", type=str, default=config.format_force, help="format_force")
    parser.add_argument("--loss_fn_id", type=int, default=config.loss_fn_id, help="loss_fn_id")
    parser.add_argument("--tblog_dir", type=str, default=config.tblog_dir, help="tblog_dir")
    parser.add_argument("--gpu", type=int, default=True, help="using gpu or not")
    args = parser.parse_args()
    args.train_length = int(args.length * 0.8)
    args.test_length = int(args.length * 0.0)
    args.val_length = args.length - args.train_length - args.test_length

    args.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")  # device = "cpu"
    # args.device = torch.device("cpu")  # device = "cpu"

    # generate_dataset("data/waterforce/", "dataset/waterforce/", config)

    print("[Step 1] Configurations")
    print("using: {}".format(args.device))
    for item in args.__dict__.items():
        if item[0][0] == "_":
            continue
        print("{}: {}".format(item[0], item[1]))

    print("[Step 2] Preparing dataset...")
    dataset_path = osp.join(args.main_path, 'dataset', args.dataset)


    # mydataset = MyDataset(root=dataset_path, split=f"{216}energy-onehot")
    # tr_dataset, v_dataset, te_dataset = split_dataset(mydataset, train_p=0.6, val_p=0.2, shuffle=True)
    #
    # train_dataset = tr_dataset
    # test_dataset = te_dataset
    # val_dataset = v_dataset


    # trainset_list = []
    # valset_list = []
    # testset_list = []
    # water1dataset = MyDataset(root=dataset_path, split=f"1-H2O")
    # fullwaterdataset = MyDataset(root=dataset_path, split=f"full-H2O")
    # water1dataset_test = MyDataset(root=dataset_path, split=f"1-H2O_test")
    # fullwaterdataset_test = MyDataset(root=dataset_path, split=f"full-H2O_test")
    # tr_dataset_1, v_dataset_1, _ = split_dataset(water1dataset_test, train_p=0.5, val_p=0.5, shuffle=True)
    # tr_dataset_2, v_dataset_2, _ = split_dataset(fullwaterdataset_test, train_p=0.5, val_p=0.5, shuffle=True)
    # trainset_list.append(water1dataset)
    # trainset_list.append(fullwaterdataset)
    # valset_list.append(v_dataset_1)
    # valset_list.append(v_dataset_2)
    # testset_list.append(tr_dataset_1)
    # testset_list.append(tr_dataset_2)
    # train_dataset = ConcatDataset(trainset_list)
    # val_dataset = ConcatDataset(valset_list)
    # test_dataset = ConcatDataset(testset_list)

    trainset_list = []
    valset_list = []
    testset_list = []
    testset_list_1water = []
    trainset_list_1water = []
    for i in range(2, 22):
        Random1dataset = MyDataset(root=dataset_path, split=f"Random1_{i}water")
        tr_data_random1, v_data_random1, te_data_random1 = split_dataset(Random1dataset, train_p=0.6, val_p=0.2, shuffle=True)
        trainset_list.append(tr_data_random1)
        testset_list.append(te_data_random1)
        valset_list.append(v_data_random1)
    for i in range(1, 22):
        Random2dataset = MyDataset(root=dataset_path, split=f"Random2_{i}water")
        tr_data_random2, v_data_random2, te_data_random2 = split_dataset(Random2dataset, train_p=0.6, val_p=0.2, shuffle=True)
        if i == 1:
            trainset_list_1water.append(tr_data_random2)
            testset_list_1water.append(te_data_random2)
        trainset_list.append(tr_data_random2)
        testset_list.append(te_data_random2)
        valset_list.append(v_data_random2)
    for i in range(2, 22):
        Random3dataset = MyDataset(root=dataset_path, split=f"Random3_{i}water")
        tr_data_random3, v_data_random3, te_data_random3 = split_dataset(Random3dataset, train_p=0.6, val_p=0.2, shuffle=True)
        trainset_list.append(tr_data_random3)
        testset_list.append(te_data_random3)
        valset_list.append(v_data_random3)
    for i in range(1, 22):
        Random4dataset = MyDataset(root=dataset_path, split=f"Random4_{i}water")
        tr_data_random4, v_data_random4, te_data_random4 = split_dataset(Random4dataset, train_p=0.8, val_p=0.1, shuffle=True)
        if i == 1:
            trainset_list_1water.append(tr_data_random4)
            testset_list_1water.append(te_data_random4)
        trainset_list.append(tr_data_random4)
        testset_list.append(te_data_random4)
        valset_list.append(v_data_random4)
    Random_1water = MyDataset(root=dataset_path, split=f"Random_1water")
    tr_data_1water, v_data_1water, te_data_1water = split_dataset(Random_1water, train_p=0.6, val_p=0.2, shuffle=True)
    trainset_list.append(tr_data_1water)
    testset_list.append(te_data_1water)
    valset_list.append(v_data_1water)
    trainset_list_1water.append(tr_data_1water)
    testset_list_1water.append(te_data_1water)
    Optimized = MyDataset(root=dataset_path, split=f"Optimized")
    trainset_list.append(Optimized)
    Optimized_test = MyDataset(root=dataset_path, split=f"Optimized_test")
    te_dataset_optimized, v_dataset_optimized, _ = split_dataset(Optimized_test, train_p=0.6, val_p=0.4, shuffle=True)
    testset_list.append(te_dataset_optimized)
    valset_list.append(v_dataset_optimized)
    train_dataset = ConcatDataset(trainset_list)
    val_dataset = ConcatDataset(valset_list)
    test_dataset = ConcatDataset(testset_list)
    test_dataset_1water = ConcatDataset(testset_list_1water)
    train_dataset_1water = ConcatDataset(trainset_list_1water)




    # trainset_list = []
    # valset_list = []
    # testset_list = []
    # testset_list_1water = []
    # trainset_list_1water = []
    # water1dataset = MyDataset(root=dataset_path, split=f"1-cufoff=1.3")
    # fullwaterdataset = MyDataset(root=dataset_path, split=f"Full-cufoff=1.3")
    # randomdataset = MyDataset(root=dataset_path, split=f"Random1-cufoff=1.3")
    # randomdataset_more = MyDataset(root=dataset_path, split=f"Random2-cufoff=1.3")
    # randomdataset_more_more = MyDataset(root=dataset_path, split=f"Random3-cufoff=1.3")
    # fullwaterdataset_test = MyDataset(root=dataset_path, split=f"Full-test-cufoff=1.3")
    # plus_1water = MyDataset(root=dataset_path, split=f"plus1-cufoff=1.3")
    # water1_5600 = MyDataset(root=dataset_path, split=f"water1_5600")
    # water2_5600 = MyDataset(root=dataset_path, split=f"water2_5600")
    # tr_data_random, v_data_random, te_data_random = split_dataset(randomdataset, train_p=0.6, val_p=0.2, shuffle=True)
    # tr_data_random_more, v_data_random_more, te_data_random_more = split_dataset(randomdataset_more, train_p=0.6, val_p=0.2, shuffle=True)
    # tr_data_random_expand, v_data_random_expand, te_data_random_expand = split_dataset(randomdataset_more_more, train_p=0.6, val_p=0.2, shuffle=True)
    # te_dataset_1, v_dataset_1, _ = split_dataset(fullwaterdataset_test, train_p=0.6, val_p=0.4, shuffle=True)
    # data1_tr, data1_v, data1_te = split_dataset(water1dataset, train_p=0.6, val_p=0.2, shuffle=True)
    # trainset_list.append(data1_tr)
    # trainset_list.append(fullwaterdataset)
    # trainset_list.append(tr_data_random)
    # trainset_list.append(tr_data_random_more)
    # trainset_list.append(tr_data_random_expand)
    # trainset_list.append(plus_1water)
    # trainset_list.append(water1_5600)
    # trainset_list.append(water2_5600)
    # valset_list.append(v_data_random)
    # valset_list.append(v_data_random_more)
    # valset_list.append(v_data_random_expand)
    # valset_list.append(v_dataset_1)
    # valset_list.append(data1_v)
    # testset_list.append(te_data_random)
    # testset_list.append(te_data_random_more)
    # testset_list.append(te_data_random_expand)
    # testset_list.append(te_dataset_1)
    # testset_list.append(data1_te)
    # testset_list_1water.append(data1_te)
    # trainset_list_1water.append(plus_1water)
    # trainset_list_1water.append(data1_tr)
    # train_dataset = ConcatDataset(trainset_list)
    # val_dataset = ConcatDataset(valset_list)
    # test_dataset = ConcatDataset(testset_list)
    # test_dataset_1water = ConcatDataset(testset_list_1water)
    # train_dataset_1water = ConcatDataset(trainset_list_1water)

    # trainset_list = []
    # valset_list = []
    # testset_list = []
    # water1dataset = MyDataset(root=dataset_path, split=f"1_more")
    # fullwaterdataset = MyDataset(root=dataset_path, split=f"Full_more")
    # randomdataset = MyDataset(root=dataset_path, split=f"random_more")
    # randomdataset_more = MyDataset(root=dataset_path, split=f"random_more_more")
    # fullwaterdataset_test = MyDataset(root=dataset_path, split=f"Full_test_more")
    # random_train, random_test, random_val = split_dataset_balance(randomdataset)
    # # tr_data_random, v_data_random, te_data_random = split_dataset(randomdataset, train_p=0.5, val_p=0.2, shuffle=True)
    # _, fullte_test, fullte_val = split_dataset_balance(fullwaterdataset_test, train_len=0, val_len=20, test_len=31)
    # # te_dataset_1, v_dataset_1, _ = split_dataset(fullwaterdataset_test, train_p=0.6, val_p=0.4, shuffle=True)
    # trainset_list.append(water1dataset)
    # trainset_list.append(fullwaterdataset)
    # trainset_list.append(random_train)
    # valset_list.append(random_val)
    # valset_list.append(fullte_val)
    # testset_list.append(random_test)
    # testset_list.append(fullte_test)
    # train_dataset = ConcatDataset(trainset_list)
    # val_dataset = ConcatDataset(valset_list)
    # test_dataset = ConcatDataset(testset_list)


    # trainset_list = []
    # valset_list = []
    # testset_list = []
    # numDatasets = 21
    # for id in range(1, numDatasets + 1):
    #     mydataset = MyDataset(root=dataset_path, split=f"{id}-H2O")
    #     tr_dataset, v_dataset, te_dataset = split_dataset(mydataset, train_p=0.6, val_p=0.2, shuffle=True)
    #     trainset_list.append(tr_dataset)
    #     valset_list.append(v_dataset)
    #     testset_list.append(te_dataset)
    #
    # train_dataset = ConcatDataset(trainset_list)
    # val_dataset = ConcatDataset(valset_list)
    # test_dataset = ConcatDataset(testset_list)

    # trainset_list = []
    # valset_list = []
    # testset_list = []
    # numDatasets = 21
    # for id in range(1, numDatasets + 1):
    #     if id == 10:
    #         continue
    #     else:
    #         mydataset = MyDataset(root=dataset_path, split=f"{id}-H2O")
    #         tr_dataset, v_dataset, te_dataset = split_dataset(mydataset, train_p=0.8, val_p=0.2, shuffle=True)
    #         trainset_list.append(tr_dataset)
    #         valset_list.append(v_dataset)
    # mydataset = MyDataset(root=dataset_path, split=f"{10}-H2O")
    # test_dataset = mydataset
    # # testset_list.append(mydataset)
    # #
    # train_dataset = ConcatDataset(trainset_list)
    # val_dataset = ConcatDataset(valset_list)
    # # test_dataset = ConcatDataset(testset_list)

    # train_dataset = MyDataset(dataset_path, subset=False, split='train')
    # test_dataset = MyDataset(dataset_path, subset=False, split='test')
    # val_dataset = MyDataset(dataset_path, subset=False, split='val')

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    test_loader_1water = DataLoader(test_dataset_1water, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    train_loader_1water = DataLoader(train_dataset_1water, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    # deg = generate_deg(train_dataset)

    print("[Step 3] Initializing model")
    main_save_path = osp.join(args.main_path, "train", args.dataset_save_as)
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    model_save_path = osp.join(main_save_path, "model_last.pt")
    initial_model_state_path = osp.join(main_save_path, "initial_state_dict.pth")
    final_model_state_path = osp.join(main_save_path, "final_state_dict.pth")
    figure_save_path_lr = osp.join(main_save_path, "lr.png")
    figure_save_path_weight = osp.join(main_save_path, "weight.png")
    figure_save_path_loss_whole = osp.join(main_save_path, "loss_whole.png")
    figure_save_path_combined = osp.join(main_save_path, "loss_combine.png")
    figure_save_path_combined_test = osp.join(main_save_path, "loss_combine_test.png")
    figure_save_path_test_loss_whole = osp.join(main_save_path, "test_loss_whole.png")
    figure_save_path_val_loss_whole = osp.join(main_save_path, "val_loss_whole.png")
    figure_save_path_loss_last_half = osp.join(main_save_path, "loss_last_half.png")
    figure_save_path_loss_last_quarter = osp.join(main_save_path, "loss_last_quarter.png")

    regression_result_train_true = osp.join(main_save_path, "all_true_ceal_no.npy")
    regression_result_train_pred = osp.join(main_save_path, "all_pred_ceal_no.npy")
    regression_result_val_true = f"{main_save_path}/val_true.npy"
    regression_result_val_pred = f"{main_save_path}/val_pred.npy"
    regression_result_test_true = f"{main_save_path}/test_true.npy"
    regression_result_test_pred = f"{main_save_path}/test_pred.npy"
    figure_regression_train_path = f"{main_save_path}/regression_train.png"
    figure_regression_val_path = f"{main_save_path}/regression_val.png"
    figure_regression_test_path = f"{main_save_path}/regression_test.png"
    # figure_regression_val_path_x = f"{main_save_path}/regression_val_x.png"
    # figure_regression_val_path_y = f"{main_save_path}/regression_val_y.png"
    # figure_regression_val_path_z = f"{main_save_path}/regression_val_z.png"
    # figure_regression_test_path_x = f"{main_save_path}/regression_test_x.png"
    # figure_regression_test_path_y = f"{main_save_path}/regression_test_y.png"
    # figure_regression_test_path_z = f"{main_save_path}/regression_test_z.png"





    print("main_save_path: {}".format(main_save_path))
    print("model_save_path: {}".format(model_save_path))
    print("figure_save_path_loss_whole: {}".format(figure_save_path_loss_whole))
    print("figure_save_path_loss_last_half: {}".format(figure_save_path_loss_last_half))
    print("figure_save_path_loss_last_quarter: {}".format(figure_save_path_loss_last_quarter))
    print("regression_result_train_true: {}".format(regression_result_train_true))
    print("regression_result_train_pred: {}".format(regression_result_train_pred))
    print("regression_result_val_true: {}".format(regression_result_val_true))
    print("regression_result_val_pred: {}".format(regression_result_val_pred))
    print("regression_result_test_true: {}".format(regression_result_test_true))
    print("regression_result_test_pred: {}".format(regression_result_test_pred))
    # print("figure_regression_train_path: {}".format(figure_regression_train_path))
    # print("figure_regression_val_path: {}".format(figure_regression_val_path))
    # print("figure_regression_test_path: {}".format(figure_regression_test_path))

    model = MyNetwork().to(args.device)
    # model.apply(init_weights)

    # 记录模型的初始参数状态
    initial_state_dict = model.state_dict()
    print(initial_state_dict)
    torch.save(initial_state_dict, initial_model_state_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=70, min_lr=0.0000001)

    #opt_adam = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=100.0)
    # summary(model, [(126, 4), (2, 324), (324,), (126,)])  # ((8064, 1), (2, 20736), 20736, 8064)
    # summary(model)
    print(model)
    print("[Step 4] Training...")

    start_time = time.time()
    start_time_0 = start_time

    epoch_loss_list = []
    epoch_test_loss_list = []
    epoch_val_loss_list = []
    epoch_test_loss_list_1water = []
    epoch_train_loss_list_1water = []
    lr_list = []
    weight_list = []
    for epoch in range(1, args.epoch + 1):
        if epoch != args.epoch:
            model, train_loss, weight = train(model, args, train_loader, optimizer)
            val_loss = test(model, args, val_loader)
            test_loss = val(model, args, test_loader)
            test_loss_1water = val(model, args, test_loader_1water)
            test_loss_1water_traindata = val(model, args, train_loader_1water)
            scheduler.step(val_loss)
            epoch_loss_list.append(train_loss)
            epoch_test_loss_list.append(test_loss)
            epoch_val_loss_list.append(val_loss)
            epoch_test_loss_list_1water.append(test_loss_1water)
            epoch_train_loss_list_1water.append(test_loss_1water_traindata)
            lr_list.append(optimizer.param_groups[0]["lr"])
            weight_list.append(weight)
        else:
            # model, train_loss, mad = train_last(model, args, train_loader, optimizer)
            model, train_loss, weight = train(model, args, train_loader, optimizer)
            val_loss = test(model, args, val_loader)
            test_loss = val(model, args, test_loader)
            test_loss_1water = val(model, args, test_loader_1water)
            test_loss_1water_traindata = val(model, args, train_loader_1water)
            scheduler.step(val_loss)
            epoch_loss_list.append(train_loss)
            epoch_test_loss_list.append(test_loss)
            epoch_val_loss_list.append(val_loss)
            epoch_test_loss_list_1water.append(test_loss_1water)
            epoch_train_loss_list_1water.append(test_loss_1water_traindata)
            lr_list.append(optimizer.param_groups[0]["lr"])
            weight_list.append(weight)
            # with open("logs/record.txt", "a") as f:
            #     f.write(mad + "\n")
        # TensorBoard
        tblog_file = os.path.join(args.tblog_dir)
        # tb_writer = SummaryWriter(tblog_file)  # 指定日志保存的目录

        # tensorboard记录
        # tb_writer.add_scalar('Loss/train', train_loss, epoch)
        # tb_writer.add_scalar('Loss/val', val_loss, epoch)
        # tb_writer.add_scalar('Lr', optimizer.param_groups[0]["lr"], epoch)
        # tb_writer.flush()
        if epoch % args.epoch_step == 0:
            now_time = time.time()
            test_log = "Epoch [{0:05d}/{1:05d}] Loss_train:{2:.6f} Loss_val:{3:.6f} Loss_test:{4:.6f} Lr:{5:.6f} (Time:{6:.6f}s Time total:{7:.2f}min Time remain: {8:.2f}min)  Weight: {9:.6f}".format(epoch, args.epoch, train_loss, val_loss, 0, optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0, (now_time - start_time_0) / 60.0 / epoch * (args.epoch - epoch), weight)
            print(test_log)
            # save_checkpoint(f'logs/checkpoint_{epoch}', model, epoch, optimizer, global_step=args.epoch)
            with open("logs/record.txt", "a") as f:
                f.write(test_log + "\n")
            start_time = now_time
            torch.save(
                {
                    "args": args,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": train_loss
                }, model_save_path)
    # 保存模型的最终状态
    final_state_dict = model.state_dict()
    # print(final_state_dict)

    # 关闭SummaryWriter
    # tb_writer.close()



    # Draw loss
    print("[Step 5] Drawing training result...")
    loss_length = len(epoch_loss_list)
    loss_x = range(1, args.epoch + 1)

    # draw weight
    draw_two_dimension(
        y_lists=[weight_list],
        x_list=loss_x,
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(weight_list[-1], min(weight_list))],
        line_style_list=["solid"],
        fig_title="Lr - whole ({} - Lr{})".format(args.dataset, args.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Weight",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_weight
    )

    # draw lr
    draw_two_dimension(
        y_lists=[lr_list],
        x_list=loss_x,
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(lr_list[-1], min(lr_list))],
        line_style_list=["solid"],
        fig_title="Lr - whole ({} - Lr{})".format(args.dataset, args.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Lr",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_lr
    )

    draw_two_dimension(
        y_lists=[
            epoch_loss_list,  # 第一条线的y值
            epoch_val_loss_list,  # 第二条线的y值
            epoch_test_loss_list,  # 第三条线的y值
            epoch_test_loss_list_1water,  # 第四条线的y值
            epoch_train_loss_list_1water
        ],
        x_list=loss_x,  # 所有线共用的x值
        color_list=["blue", "red", "green", "purple", "orange"],  # 指定每条线的颜色
        legend_list=["Train Loss", "Validation Loss", "Test Loss", "Test Loss 1-Water", "Test Loss 1-Water train"],  # 为每条线添加图例
        line_style_list=["solid", "dashed", "dotted", "dashdot", "solid"],  # 指定每条线的样式
        fig_title="Train and Validation Loss ({} - Loss{})".format(args.dataset, args.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_combined_test  # 保存组合图
    )

    draw_two_dimension(
        y_lists=[epoch_loss_list, epoch_val_loss_list],
        x_list=loss_x,
        color_list=["blue", "red"],  # You can specify colors for each curve
        legend_list=["Train Loss", "Validation Loss"],  # Add legends for each curve
        line_style_list=["solid", "dashed"],  # You can specify line styles
        fig_title="Train and Validation Loss ({} - Loss{})".format(args.dataset, args.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_combined  # Save the combined plot
    )

    # # draw test loss_whole
    # draw_two_dimension(
    #     y_lists=[epoch_test_loss_list],
    #     x_list=loss_x,
    #     color_list=["blue"],
    #     legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_test_loss_list[-1], min(epoch_test_loss_list))],
    #     line_style_list=["solid"],
    #     fig_title="Loss - whole ({} - Loss{})".format(args.dataset, args.loss_fn_id),
    #     fig_x_label="Epoch",
    #     fig_y_label="Loss",
    #     fig_size=(8, 6),
    #     show_flag=False,
    #     save_flag=True,
    #     save_path=figure_save_path_test_loss_whole
    # )

    # draw val loss_whole
    draw_two_dimension(
        y_lists=[epoch_val_loss_list],
        x_list=loss_x,
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_val_loss_list[-1], min(epoch_val_loss_list))],
        line_style_list=["solid"],
        fig_title="Loss - whole ({} - Loss{})".format(args.dataset, args.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_val_loss_whole
    )

    # draw train loss_whole
    draw_two_dimension(
        y_lists=[epoch_loss_list],
        x_list=loss_x,
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_loss_list[-1], min(epoch_loss_list))],
        line_style_list=["solid"],
        fig_title="Loss - whole ({} - Loss{})".format(args.dataset, args.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_loss_whole
    )


    # draw loss_last_half
    draw_two_dimension(
        y_lists=[epoch_loss_list[-(loss_length // 2):]],
        x_list=loss_x[-(loss_length // 2):],
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_loss_list[-1], min(epoch_loss_list))],
        line_style_list=["solid"],
        fig_title="Loss - last half ({} - Loss{})".format(args.dataset, args.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_loss_last_half
    )

    # draw loss_last_quarter
    draw_two_dimension(
        y_lists=[epoch_loss_list[-(loss_length // 4):]],
        x_list=loss_x[-(loss_length // 4):],
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_loss_list[-1], min(epoch_loss_list))],
        line_style_list=["solid"],
        fig_title="Loss - last quarter ({} - Loss{})".format(args.dataset, args.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_loss_last_quarter
    )

    # Test
    print("[Step 6] Testing...")
    model.eval()
    train_true_list_energy = []
    train_pred_list_energy = []
    val_true_list_energy = []
    val_pred_list_energy = []
    test_true_list_energy = []
    test_pred_list_energy = []
    train_true_list_force = []
    train_pred_list_force = []
    val_true_list_force = []
    val_pred_list_force = []
    test_true_list_force = []
    test_pred_list_force = []
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2,
                                  worker_init_fn=worker_init_fn, generator=g, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, worker_init_fn=worker_init_fn,
                                 generator=g, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, worker_init_fn=worker_init_fn,
                                 generator=g, shuffle=True)

    total_error = 0
    for data in train_dataloader:
        data = data.to(args.device)
        input_dict = dict({
            "x": data.x,
            "edge_index": data.edge_index,
            "edge_attr": data.edge_attr,
            "batch": data.batch
        })

        #out = model(input_dict)
        # loss=(((out.squeeze()-data.y)**2).mean(1).sqrt()).mean()
        #out = model(data.x, data.edge_index, data.edge_attr)
        mad_flag = False
        with torch.no_grad():
            # out, _ = model(input_dict, mad_flag)
            out, weight = model(input_dict, mad_flag)
        print(out)
        # train_true_list_energy += list(reverse_min_max_scaler_1d(data.y.cpu().detach().numpy()))
        train_true_list_energy += list(data.y.cpu().detach().numpy())
        train_true_list_force += list(data.z.cpu().detach().numpy())

        # print(reverse_min_max_scaler_1d(out["energy"].squeeze().cpu().detach().numpy()))
        print(out["energy"].squeeze().cpu().detach().numpy())
        # train_pred_list_energy += reverse_min_max_scaler_1d(out["energy"].squeeze().cpu().detach().numpy()).tolist()
        train_pred_list_energy += list((out["energy"].squeeze().cpu().detach().numpy()))
        train_pred_list_force += list(out["force"].squeeze().cpu().detach().numpy())
        total_error += 1.0 * (reverse_min_max_scaler_1d(out["energy"].squeeze()) - reverse_min_max_scaler_1d(data.y)).abs().mean()
        # total_error += 1.0 * (out["energy"].squeeze() - data.y).abs().mean()

        # total_error += 1.0 * (((out["force"].squeeze() - data.z) ** 2).mean(1).sqrt()).mean()
        # total_error += 0.5 * (((out["force"].squeeze() - data.z) ** 2).mean(1).sqrt()).mean() + 0.5 * (
        #            out["energy"].squeeze() - data.y).abs().mean()
    train_loss = total_error / len(train_dataloader.dataset)
    print(train_loss)
    print("ok")

    # total_error = 0
    # for data in val_dataloader:
    #     data = data.to(args.device)
    #     input_dict = dict({
    #         "x": data.x,
    #         "edge_index": data.edge_index,
    #         "edge_attr": data.edge_attr,
    #         "batch": data.batch
    #     })
    #     #out = model(input_dict)
    #     #out = model(data.x, data.edge_index,  data.edge_attr)
    #     out, _ = model(input_dict, mad_flag)
    #
    #     # val_true_list_energy += list(reverse_min_max_scaler_1d(data.y.cpu().detach().numpy()))
    #     val_true_list_energy += list(data.y.cpu().detach().numpy())
    #     val_true_list_force += list(data.z.cpu().detach().numpy())
    #     # val_pred_list_energy += reverse_min_max_scaler_1d(out["energy"].squeeze().cpu().detach().numpy()).tolist()
    #     val_pred_list_energy += list(out["energy"].squeeze().cpu().detach().numpy())
    #     val_pred_list_force += list(out["force"].squeeze().cpu().detach().numpy())
    #
    #     # total_error += 1.0 * (reverse_min_max_scaler_1d(out["energy"].squeeze()) - reverse_min_max_scaler_1d(data.y)).abs().mean()
    #     total_error += 1.0 * (out["energy"].squeeze() - data.y).abs().mean()
    #     # total_error += 1.0 * (((out["force"].squeeze() - data.z) ** 2).mean(1).sqrt()).mean()
    #     # total_error += 0.5 * (((out["force"].squeeze() - data.z) ** 2).mean(1).sqrt()).mean() + 0.5 * (
    #     #             out["energy"].squeeze() - data.y).abs().mean()
    # val_loss = total_error / len(val_dataloader.dataset)
    # print(val_loss)

    total_error = 0
    for data in test_dataloader:
        data = data.to(args.device)
        input_dict = dict({
            "x": data.x,
            "edge_index": data.edge_index,
            "edge_attr": data.edge_attr,
            "batch": data.batch
        })
        #out = model(input_dict)
        with torch.no_grad():
            # out, _ = model(input_dict, mad_flag)
            out, weight = model(input_dict, mad_flag)
        test_true_list_energy += list(data.y.cpu().detach().numpy())
        test_true_list_force += list(data.z.cpu().detach().numpy())
        test_pred_list_energy += list(out["energy"].squeeze().cpu().detach().numpy())
        test_pred_list_force += list(out["force"].squeeze().cpu().detach().numpy())
        total_error += 1.0 * (
                    reverse_min_max_scaler_1d(out["energy"].squeeze()) - reverse_min_max_scaler_1d(data.y)).abs().mean()
        # total_error += 1.0 * (out["energy"].squeeze() - data.y).abs().mean()
        # total_error += 1.0 * (((out["force"].squeeze() - data.z) ** 2).mean(1).sqrt()).mean()
        # total_error += 0.5 * (((out["force"].squeeze() - data.z) ** 2).mean(1).sqrt()).mean() + 0.5 * (
        #             out["energy"].squeeze() - data.y).abs().mean()
    test_loss = total_error / len(test_dataloader.dataset)
    print(test_loss)
    print("weight:", weight)

    train_true_list_energy = np.asarray(train_true_list_energy)
    train_pred_list_energy = np.asarray(train_pred_list_energy)
    # print("Max values:", np.max(train_true_list_energy))
    # print("Min values:", np.min(train_true_list_energy))
    # val_true_list_energy = np.asarray(val_true_list_energy)
    # val_pred_list_energy = np.asarray(val_pred_list_energy)
    test_true_list_energy = np.asarray(test_true_list_energy)
    test_pred_list_energy = np.asarray(test_pred_list_energy)
    # print("Max values:", np.max(test_true_list_energy))
    # print("Min values:", np.min(test_true_list_energy))
    train_true_list_force = np.asarray(train_true_list_force)
    train_pred_list_force = np.asarray(train_pred_list_force)
    val_true_list_force = np.asarray(val_true_list_force)
    val_pred_list_force = np.asarray(val_pred_list_force)
    test_true_list_force = np.asarray(test_true_list_force)
    test_pred_list_force = np.asarray(test_pred_list_force)

    np.save(regression_result_train_true, train_true_list_energy)
    np.save(regression_result_train_pred, train_pred_list_energy)
    # np.save(regression_result_val_true, val_true_list_energy)
    # np.save(regression_result_val_pred, val_pred_list_energy)
    np.save(regression_result_test_true, test_true_list_energy)
    np.save(regression_result_test_pred, test_pred_list_energy)

    final_state_dict = model.state_dict()
    # print(final_state_dict)
    torch.save(final_state_dict, final_model_state_path)

    print("[Step 7] Drawing train/val/test result...")

    # writer = SummaryWriter()
    # writer.add_scalar('logs', loss, n_iter)
    # r_train = compute_correlation(train_true_list_energy, train_pred_list_energy)
    # r_train = float(r_train)
    # draw_two_dimension_regression(
    #     x_lists=[train_true_list_energy],
    #     y_lists=[train_pred_list_energy],
    #     color_list=["red"],
    #     legend_list=["Regression: R^2={0:.3f}".format(r_train ** 2.0)],
    #     line_style_list=["solid"],
    #     fig_title="Regression - {0} - Train - {1} points".format(args.dataset, len(train_true_list_energy)),
    #     fig_x_label="Truth",
    #     fig_y_label="Predict",
    #     fig_size=(8, 6),
    #     show_flag=False,
    #     save_flag=True,
    #     save_path=figure_regression_train_path
    # )
    #
    # r_val = compute_correlation(val_true_list_energy, val_pred_list_energy)
    # draw_two_dimension_regression(
    #     x_lists=[val_true_list_energy],
    #     y_lists=[val_pred_list_energy],
    #     color_list=["red"],
    #     legend_list=["Regression: R^2={0:.3f}".format(r_val ** 2.0)],
    #     line_style_list=["solid"],
    #     fig_title="Regression - {0} - Val - {1} points".format(args.dataset, len(val_true_list_energy)),
    #     fig_x_label="Truth",
    #     fig_y_label="Predict",
    #     fig_size=(8, 6),
    #     show_flag=False,
    #     save_flag=True,
    #     save_path=figure_regression_val_path
    # )
    #
    # r_test = compute_correlation(test_true_list_energy, test_pred_list_energy)
    # draw_two_dimension_regression(
    #     x_lists=[test_true_list_energy],
    #     y_lists=[test_pred_list_energy],
    #     color_list=["red"],
    #     legend_list=["Regression: R^2={0:.3f}".format(r_test ** 2.0)],
    #     line_style_list=["solid"],
    #     fig_title="Regression - {0} - Test - {1} points".format(args.dataset, len(test_true_list_energy)),
    #     fig_x_label="Truth",
    #     fig_y_label="Predict",
    #     fig_size=(8, 6),
    #     show_flag=False,
    #     save_flag=True,
    #     save_path=figure_regression_test_path
    # )
    # print(mad)


    # r_test = compute_correlation(test_true_list_force[:,0], test_pred_list_force[:,0])
    # draw_two_dimension_regression(
    #     x_lists=[test_true_list_force[:,0]],
    #     y_lists=[test_pred_list_force[:,0]],
    #     color_list=["red"],
    #     legend_list=["Regression: R^2={0:.3f}".format(r_test ** 2.0)],
    #     line_style_list=["solid"],
    #     fig_title="Regression - {0} - Test x - {1} points".format(args.dataset, len(test_true_list_energy)),
    #     fig_x_label="Truth",
    #     fig_y_label="Predict",
    #     fig_size=(8, 6),
    #     show_flag=False,
    #     save_flag=True,
    #     save_path=figure_regression_test_path_x
    # )
    #
    # r_test = compute_correlation(test_true_list_force[:, 1], test_pred_list_force[:, 1])
    # draw_two_dimension_regression(
    #     x_lists=[test_true_list_force[:, 1]],
    #     y_lists=[test_pred_list_force[:, 1]],
    #     color_list=["red"],
    #     legend_list=["Regression: R^2={0:.3f}".format(r_test ** 2.0)],
    #     line_style_list=["solid"],
    #     fig_title="Regression - {0} - Test y - {1} points".format(args.dataset, len(test_true_list_energy)),
    #     fig_x_label="Truth",
    #     fig_y_label="Predict",
    #     fig_size=(8, 6),
    #     show_flag=False,
    #     save_flag=True,
    #     save_path=figure_regression_test_path_y
    # )
    #
    # r_test = compute_correlation(test_true_list_force[:, 2], test_pred_list_force[:, 2])
    # draw_two_dimension_regression(
    #     x_lists=[test_true_list_force[:, 2]],
    #     y_lists=[test_pred_list_force[:, 2]],
    #     color_list=["red"],
    #     legend_list=["Regression: R^2={0:.3f}".format(r_test ** 2.0)],
    #     line_style_list=["solid"],
    #     fig_title="Regression - {0} - Test z - {1} points".format(args.dataset, len(test_true_list_energy)),
    #     fig_x_label="Truth",
    #     fig_y_label="Predict",
    #     fig_size=(8, 6),
    #     show_flag=False,
    #     save_flag=True,
    #     save_path=figure_regression_test_path_z
    # )
    #
    # r_val = compute_correlation(val_true_list_force[:, 0], val_pred_list_force[:, 0])
    # draw_two_dimension_regression(
    #     x_lists=[val_true_list_force[:, 0]],
    #     y_lists=[val_pred_list_force[:, 0]],
    #     color_list=["red"],
    #     legend_list=["Regression: R^2={0:.3f}".format(r_val ** 2.0)],
    #     line_style_list=["solid"],
    #     fig_title="Regression - {0} - Val x - {1} points".format(args.dataset, len(val_true_list_energy)),
    #     fig_x_label="Truth",
    #     fig_y_label="Predict",
    #     fig_size=(8, 6),
    #     show_flag=False,
    #     save_flag=True,
    #     save_path=figure_regression_val_path_x
    # )
    #
    # r_val = compute_correlation(val_true_list_force[:, 1], val_pred_list_force[:, 1])
    # draw_two_dimension_regression(
    #     x_lists=[val_true_list_force[:, 1]],
    #     y_lists=[val_pred_list_force[:, 1]],
    #     color_list=["red"],
    #     legend_list=["Regression: R^2={0:.3f}".format(r_val ** 2.0)],
    #     line_style_list=["solid"],
    #     fig_title="Regression - {0} - Val y - {1} points".format(args.dataset, len(val_true_list_energy)),
    #     fig_x_label="Truth",
    #     fig_y_label="Predict",
    #     fig_size=(8, 6),
    #     show_flag=False,
    #     save_flag=True,
    #     save_path=figure_regression_val_path_y
    # )
    #
    # r_val = compute_correlation(val_true_list_force[:, 2], val_pred_list_force[:, 2])
    # draw_two_dimension_regression(
    #     x_lists=[val_true_list_force[:, 2]],
    #     y_lists=[val_pred_list_force[:, 2]],
    #     color_list=["red"],
    #     legend_list=["Regression: R^2={0:.3f}".format(r_val ** 2.0)],
    #     line_style_list=["solid"],
    #     fig_title="Regression - {0} - Val z - {1} points".format(args.dataset, len(val_true_list_energy)),
    #     fig_x_label="Truth",
    #     fig_y_label="Predict",
    #     fig_size=(8, 6),
    #     show_flag=False,
    #     save_flag=True,
    #     save_path=figure_regression_val_path_z
    # )



    # model.set_cp2k_flag(True)
    # example_input = dict({
    #     "atom_info": torch.tensor(np.array([[8, 10.6666846352, 7.7624477189, 10.7266882343],
    #                                         [1, 10.4454189356, 6.8752349577, 11.0388329543],
    #                                         [1, 10.3223695991, 7.8187995109, 9.8162635102]], dtype=np.float32))
    # })
    # # Run the model once to trace the computational graph
    # # traced_module = torch.jit.script(model)
    # # traced_module = torch.jit.trace(model, example_input, strict=False)
    #
    # # Save the model to a .pth file
    # # torch.jit.save(traced_module, 'model_force.pth')
    # loaded = torch.jit.load('model_energy.pth')
    # traced_module = torch.jit.trace(loaded, example_input, strict=False)



if __name__ == "__main__":
    run()
    # a = np.load("train/GCN_N3P/test_pred.npy")
    # b = np.load("train/GCN_N3P/test_true.npy")
    # for aa, bb in zip(a, b):
    #     print(aa, bb)
    # with open("dataset/GCN_N3P/raw/train.pickle", "rb") as f:
    #     data = pickle.load(f)
    # for i in range(len(data) - 1)[:10]:
    #
    #     print(i, np.sum(np.abs(data[i]["bond_type"].cpu().detach().numpy() - data[i + 1]["bond_type"].cpu().detach().numpy())))
    #     np.set_printoptions(threshold=np.inf)
    #     print(data[i]["bond_type"].cpu().detach().numpy())

