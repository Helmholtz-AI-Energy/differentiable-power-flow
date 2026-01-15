import pandas as pd

def main():
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    case_names = ["case118", "case_illinois200", "case300", "case1354pegase", "case1888rte",
                  "case2869pegase", "case3120sp", "case6495rte", "case6515rte", "case9241pegase"]

    for case_name in case_names:
        df = pd.read_csv(
            f"out/temp/ex10_trials_{case_name}.csv")
        print(case_name, df.sort_values("value")["value"].iloc[0])
        print(case_name, df.sort_values("value").iloc[0])
        print(" ")
        print(" ")

if __name__ == "__main__":
    main()
