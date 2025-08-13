import matplotlib.pyplot as plt
import pandas as pd


class PlotQTable:
    def __init__(self, q_table_path: str):
        self._q_table = pd.read_csv(q_table_path, index_col=0)

    def plot_q_table(self):
        plt.figure(figsize=(25, 20))

        plt.subplot(3, 1, 1)
        plt.plot(self._q_table.iloc[:22, 2], label="1->1")
        plt.plot(self._q_table.iloc[:22, 1], label="1->0")
        plt.plot(self._q_table.iloc[:22, 0], label="1->-1")
        plt.legend()
        plt.title("holding = 1")
        plt.xlabel("state index")

        plt.subplot(3, 1, 2)
        plt.plot(self._q_table.iloc[22:44, 2], label="0->1")
        plt.plot(self._q_table.iloc[22:44, 1], label="0->0")
        plt.plot(self._q_table.iloc[22:44, 0], label="0->-1")
        plt.legend()
        plt.title("holding = 0")
        plt.xlabel("state index")

        plt.subplot(3, 1, 3)
        plt.plot(self._q_table.iloc[44:, 2], label="-1->1")
        plt.plot(self._q_table.iloc[44:, 1], label="-1->0")
        plt.plot(self._q_table.iloc[44:, 0], label="-1->-1")
        plt.legend()
        plt.title("holding = -1")
        plt.xlabel("state index")

        plt.savefig("./figure/q_table.png")


if __name__ == "__main__":
    plot_q_table = PlotQTable("./q_table.csv")
    plot_q_table.plot_q_table()
