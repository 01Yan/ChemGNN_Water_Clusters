import time
# class Config:
#     main_path = "./"
#     dataset = "waterforce"
#     model="GNEW"
#     max_natoms = 63
#     length = 39397
#     train_length = int(length * 0.6)
#     test_length = int(length * 0.3)
#     val_length = length - train_length - test_length
#     root_bmat = main_path + 'data/{}/BTMATRIX'.format(dataset)
#     root_dmat = main_path + 'data/{}/DMATRIXES'.format(dataset)
#     root_conf = main_path + 'data/{}/CONFIG'.format(dataset)
#     root_force = main_path + 'data/{}/FORCE'.format(dataset)
#     format_bmat = "output_CONFIG_{}.xyz"
#     format_dmat = "output_CONFIG_{}.xyz"
#     format_conf = "CONFIG_{}.xyz"
#     format_force = "MLFORCE_{}"
#     format_eigen = "MLENERGY_{}"
#
#     tblog_dir = "logs"
#     #format_charge = "CHARGES/charge_data_{}"
#     loss_fn_id = 1
#     threshold = -17.2
#
#     epoch = 500
#     epoch_step = 1  # print loss every {epoch_step} epochs
#     batch_size= 64
#     lr = 0.0001
#     seed = 0

class Config:
    main_path = "./"
    dataset = "waterforce"
    model="GNEW"
    max_natoms = 3
    length = 2319
    train_length = int(length * 0.6)
    test_length = int(length * 0.2)
    val_length = length - train_length - test_length
    root_bmat = main_path + 'data/{}/BTMATRIXES'.format(dataset)
    root_dmat = main_path + 'data/{}/DMATRIXES'.format(dataset)
    root_conf = main_path + 'data/{}/CONFIGS'.format(dataset)
    root_force = main_path + 'data/{}/AIFORCE'.format(dataset)
    format_bmat = "BTMATRIX_{}"
    format_dmat = "DMATRIX_{}"
    format_conf = "CONFIG_{}"
    format_force = "FORCE_{}"
    format_eigen = "ENERGY_{}"
    #format_charge = "CHARGES/charge_data_{}"

    tblog_dir = "logs"
    loss_fn_id = 1
    threshold = -17.2

    epoch = 2000
    epoch_step = 1  # print loss every {epoch_step} epochs
    batch_size= 128
    lr = 0.0001 #tiao xiao le yg shuliangji
    seed = int(time.time())
    weight = 20.0

# class Config:
#     main_path = "./"
#     dataset = "waterforce"
#     model="GNEW"
#     max_natoms = 648
#     length = 5076
#     train_length = int(length * 0.6)
#     test_length = int(length * 0.2)
#     val_length = length - train_length - test_length
#     root_bmat = main_path + 'data/{}/BTMATRIXES'.format(dataset)
#     root_dmat = main_path + 'data/{}/DMATRIXES'.format(dataset)
#     root_conf = main_path + 'data/{}/CONFIG'.format(dataset)
#     root_force = main_path + 'data/{}/FORCE'.format(dataset)
#     format_bmat = "BTMATRIX_{}"
#     format_dmat = "DTMATRIX_{}"
#     format_conf = "water{}"
#     format_force = "MLFORCE_{}"
#     format_eigen = "MLENERGY_{}"
#     #format_charge = "CHARGES/charge_data_{}"
#     loss_fn_id = 1
#     threshold = -17.2
#     tblog_dir = "logs"
# #
#     epoch = 600
#     epoch_step = 1  # print loss every {epoch_step} epochs
#     batch_size= 36
#     lr = 0.00001
#     seed = 10086

class Config_dataset:
    main_path = "./"
    dataset = "data_1-21-H2O"
    # length = 5000
    length = 1001
    root_bmat = main_path + 'data/{}/BTMATRIXES'.format(dataset)
    root_dmat = main_path + 'data/{}/DMATRIXES'.format(dataset)
    root_conf = main_path + 'data/{}/CONFIG'.format(dataset)
    root_force = main_path + 'data/{}/FORCE'.format(dataset)
    format_bmat = "BTMATRIX_{}"
    format_dmat = "DMATRIX_{}"
    format_conf = "water{}"
    format_force = "MLFORCE_{}"
    format_eigen = "MLENERGY_{}"

class Config_dataset2:
    main_path = "./"
    dataset = "data_1-21-H2O"
    # length = 5000
    length = 1001
    root_bmat = main_path + 'data/{}/BTMATRIXES'.format(dataset)
    root_dmat = main_path + 'data/{}/DMATRIXES'.format(dataset)
    root_conf = main_path + 'data/{}/CONFIG'.format(dataset)
    root_force = main_path + 'data/{}/FORCE'.format(dataset)
    format_bmat = "BTMATRIX_{}"
    format_dmat = "DMATRIX_{}"
    format_conf = "water{}"
    format_force = "MLFORCE_{}"
    format_eigen = "MLENERGY_{}"

class Config_dataset3:
    main_path = "./"
    dataset = "data_1-21-H2O"
    # length = 5000
    length = 3003
    root_bmat = main_path + 'data/{}/BTMATRIXES'.format(dataset)
    root_dmat = main_path + 'data/{}/DMATRIXES'.format(dataset)
    root_conf = main_path + 'data/{}/CONFIG'.format(dataset)
    root_force = main_path + 'data/{}/FORCE'.format(dataset)
    format_bmat = "BTMATRIX_{}"
    format_dmat = "DMATRIX_{}"
    format_conf = "water{}"
    format_force = "MLFORCE_{}"
    format_eigen = "MLENERGY_{}"

class Config_dataset4:
    main_path = "./"
    dataset = "data_1-21-H2O"
    # length = 5000
    length = 23746
    root_bmat = main_path + 'data/{}/BTMATRIXES'.format(dataset)
    root_dmat = main_path + 'data/{}/DMATRIXES'.format(dataset)
    root_conf = main_path + 'data/{}/CONFIG'.format(dataset)
    root_force = main_path + 'data/{}/FORCE'.format(dataset)
    format_bmat = "BTMATRIX_{}"
    format_dmat = "DMATRIX_{}"
    format_conf = "water{}"
    format_force = "MLFORCE_{}"
    format_eigen = "MLENERGY_{}"

class Config_dataset5:
    main_path = "./"
    dataset = "data_1-21-H2O"
    # length = 5000
    length = 1020
    root_bmat = main_path + 'data/{}/BTMATRIXES'.format(dataset)
    root_dmat = main_path + 'data/{}/DMATRIXES'.format(dataset)
    root_conf = main_path + 'data/{}/CONFIG'.format(dataset)
    root_force = main_path + 'data/{}/FORCE'.format(dataset)
    format_bmat = "BTMATRIX_{}"
    format_dmat = "DMATRIX_{}"
    format_conf = "water{}"
    format_force = "MLFORCE_{}"
    format_eigen = "MLENERGY_{}"

class Config_dataset6:
    main_path = "./"
    dataset = "data_1-21-H2O"
    # length = 5000
    length = 20020
    root_bmat = main_path + 'data/{}/BTMATRIXES'.format(dataset)
    root_dmat = main_path + 'data/{}/DMATRIXES'.format(dataset)
    root_conf = main_path + 'data/{}/CONFIG'.format(dataset)
    root_force = main_path + 'data/{}/FORCE'.format(dataset)
    format_bmat = "BTMATRIX_{}"
    format_dmat = "DMATRIX_{}"
    format_conf = "water{}"
    format_force = "MLFORCE_{}"
    format_eigen = "MLENERGY_{}"

config = Config
config_data = Config_dataset
config_data2 = Config_dataset2
config_data3 = Config_dataset3
config_data4 = Config_dataset4
config_data5 = Config_dataset5
config_data6 = Config_dataset6

