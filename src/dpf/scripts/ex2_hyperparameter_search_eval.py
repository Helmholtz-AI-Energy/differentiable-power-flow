import pandas as pd


def main():
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)

    optimizer_strats = ["Adam", "SGD", "RMSprop"]
    # optimizer_strats = ["Adam"]

    scheduler_strats = ["constant", "StepLR", "ReduceLROnPlateau", "MultiStepLR"]
    # scheduler_strats = ["constant"]

    num_iters = [50]

    print("printing values for each optimizer and scheduler")
    for optimizer_strat in optimizer_strats:
        for scheduler_strat in scheduler_strats:
            for num_iter in num_iters:
                df = pd.read_csv(
                    f"out/temp/ex2_trials_{optimizer_strat}_{scheduler_strat}_{num_iter}.csv"
                )
                print(
                    optimizer_strat,
                    scheduler_strat,
                    df.sort_values("value")["value"].iloc[0],
                )

    # Adam constant 0.1967925840098253
    # Adam StepLR 0.1994482126301495
    # Adam ReduceLROnPlateau 0.1962377621975704
    # Adam MultiStepLR 0.2046412473571692
    # SGD constant 9214.154891490169
    # SGD StepLR 9214.154891490169
    # SGD ReduceLROnPlateau 9214.154891490169
    # SGD MultiStepLR 9214.154891490169
    # RMSprop constant 0.1951365592682966
    # RMSprop StepLR 0.0197887050869326                 **best**
    # RMSprop ReduceLROnPlateau 0.0850413949614633
    # RMSprop MultiStepLR 0.0695452015602939
    print(" ")

    print("best adam parameters:")

    df_adam_50 = pd.read_csv("out/temp/ex2_trials_Adam_constant_50.csv")
    print(df_adam_50.sort_values("value").iloc[0])

    print(" ")
    print("longer run: ")

    df_adam_constant_1000 = pd.read_csv("out/temp/ex2_trials_Adam_constant_1000.csv")
    print(df_adam_constant_1000.sort_values("value").iloc[0])

    ##########
    # best RMSprop params

    # params_alpha           0.361584
    # params_cooldown              63
    # params_factor          0.145166
    # params_loss_fn               L1
    # params_lr              0.004736
    # params_momentum        0.962065
    # params_patience              67
    # params_threshold       0.024118
    # params_weight_decay    0.927735
    # value                  0.014152


if __name__ == "__main__":
    main()
