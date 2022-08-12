
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import requests
import sys


def download_dataset(link, filename):
    response = requests.get(link)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print("File has been downloaded")
    else:
        print("File has not been downloaded", file=sys.stderr)
        print("Status code:", response.status_code, file=sys.stderr)


if __name__ == '__main__':
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN" \
          "-SkillsNetwork/Data%20Files/Canada.xlsx"
    excel_name = "Canada.xlsx"
    download_dataset(url, excel_name)

    df = pd.read_excel(excel_name, sheet_name="Canada by Citizenship", skiprows=range(20), skipfooter=2)
    df.drop(["Type", "Coverage", "AREA", "REG", "DEV"], axis=1, inplace=True)
    df.rename(columns={"OdName": "Country", "AreaName": "Continent", "RegName": "Region"}, inplace=True)
    years = list(range(1980, 2014))
    df["Total"] = df[years].sum(axis=1)
    df.set_index("Country", inplace=True)
    df.rename(index={"United Kingdom of Great Britain and Northern Ireland": "United Kingdom"}, inplace=True)
    mpl.style.use(["ggplot"])

    russia = df.loc["Russian Federation", years]
    japan = df.loc["Japan", years].transpose()
    fig = plt.figure()
    axr = fig.add_subplot(1, 2, 1)  # add subplot 1 (1 row, 2 columns, first plot)
    axj = fig.add_subplot(1, 2, 2)  # add subplot 2 (1 row, 2 columns, second plot)

    russia.plot(kind="line", color="blue", figsize=(20, 6), ax=axr)
    axr.set_title("Immigration from Russia")
    axr.set_xlabel("Years")
    axr.set_ylabel("Number of immigrants")
    axr.set_xlim([1979, 2014])
    axr.text(0.06, 0.08, "1991 Dissolution of the USSR", transform=axr.transAxes)

    japan.plot(kind="line", color="red", figsize=(20, 6), ax=axj)
    axj.set_title("Immigration from Japan")
    axj.set_ylabel("Number of immigrants")
    axj.set_xlabel("Years")
    axj.set_xlim([1979, 2014])
    axj.set_ylim([0, 1400])
    axj.text(0.18, 0.08, "1985 Plaza Accord", transform=axj.transAxes)
    plt.savefig("1_Russia_Japan.png")
    plt.show()

    fig = plt.figure()
    fig.suptitle("Compare migration between neighboring countries")
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    df_CI = df.loc[["China", "India"], years].transpose()
    df_CI.plot(kind="line", ax=ax1)
    df_DNS = df.loc[["Norway", "Sweden", "Denmark"], years].transpose()
    df_DNS.plot(kind="line", ax=ax2)
    df_LLE = df.loc[["Lithuania", "Latvia", "Estonia"], years].transpose()
    df_LLE.plot(kind="line", ax=ax3)
    df_RFU = df.loc[["Russian Federation", "Ukraine"], years].transpose()
    df_RFU.plot(kind="line", figsize=(9, 6), ax=ax4)
    plt.savefig("2_neighboring_countries.png")
    plt.show()

    df_sort = df.sort_values("Total", ascending=False)
    df_top5 = df_sort.head(5)[years].transpose()
    df_top5.plot(kind="area", alpha=0.55, figsize=(8, 6))
    plt.xlim(1990, 2013)
    plt.title("Immigration trend of top 5 countries")
    plt.ylabel("Number of Immigrants")
    plt.xlabel("Years")
    plt.savefig("3_top5_countries.png")
    plt.show()

    df_iceland = df.loc["Iceland", years]
    df_iceland.plot(kind="bar", figsize=(9, 6), rot=45)
    plt.title("Immigration from Iceland")
    plt.ylabel("Number of Immigrants")
    plt.annotate("", xy=(32, 70), xytext=(28, 20), xycoords='data',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
    plt.annotate('2008 - 2011 Financial Crisis', xy=(28, 30), rotation=72.5, va='bottom', ha='left')
    plt.savefig("4_Iceland.png")
    plt.show()

    df_continents = df.groupby("Continent", axis=0).sum()
    colors_list = ["lightcoral", "yellowgreen", "lightskyblue", "gold", "lightgreen", "pink"]
    df_continents["Total"].plot(kind="pie", figsize=(7, 6), autopct="%1.1f%%", startangle=-25,
                                pctdistance=0.8, colors=colors_list)
    plt.title("Immigration to Canada by continent 1980-2013")
    plt.ylabel("")
    plt.savefig("5_continents.png")
    plt.show()

    df_tot = pd.DataFrame(df[years].sum(axis=0))
    df_tot.reset_index(inplace=True)
    df_tot.columns = ["year", "total"]

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    sns.regplot(x="year", y="total", data=df_tot, scatter_kws={"color": "darkblue"}, line_kws={"color": "red"})
    plt.title("Total number of immigrants")
    plt.ylabel("Number of immigrants")
    plt.xlabel("Years")
    plt.ylim(0, 300000)
    sns.despine()
    plt.savefig("6_immigration_trend.png")
    plt.show()

    df_years = df[years].transpose()
    df_years.index.name = "Year"
    df_years.reset_index(inplace=True)
    norm_brazil = (df_years["Brazil"] - df_years["Brazil"].min()) / \
                  (df_years["Brazil"].max() - df_years["Brazil"].min())
    norm_argentina = (df_years["Argentina"] - df_years["Argentina"].min()) / \
                     (df_years["Argentina"].max() - df_years["Argentina"].min())

    ax1 = df_years.plot(kind="scatter", x="Year", y="Brazil", xlim=(1979, 2015), color="green", alpha=0.75,
                        s=norm_brazil*200)
    ax2 = df_years.plot(kind="scatter", x="Year", y="Argentina", xlim=(1979, 2015), color="blue", alpha=0.75,
                        s=norm_argentina*200, ax=ax1)
    ax1.set_ylabel("Number of Immigrants")
    ax1.set_title("Immigration from Brazil and Argentina from 1980 to 2013")
    ax1.legend(["Brazil", "Argentina"], loc="upper left", fontsize="x-large")
    plt.savefig("7_Brazil_Argentina.png")
    plt.show()

    df_dns = df.loc[["Denmark", "Norway", "Sweden"], years]
    df_DNS_total = pd.DataFrame(df_dns.sum(axis=0))
    df_DNS_total.reset_index(inplace=True)
    df_DNS_total.columns = ["year", "total"]
    total_values = df_DNS_total["total"].sum()
    country_proportions = df_dns.sum(axis=1) / total_values
    width = 40
    height = 10
    tiles_country = (country_proportions * width * height).round().astype(int)
    prefix_sum = np.cumsum(tiles_country)
    waffle_chart = np.zeros((height, width), dtype=np.uint)
    category_index = 0
    for col in range(width):
        for row in range(height):
            if col*height+row+1 > prefix_sum[category_index]:
                category_index += 1
            waffle_chart[row, col] = category_index

    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    plt.xticks([])
    plt.yticks([])
    legend_handles = []
    for i, category in enumerate(tiles_country.index.values):
        label_str = category + ' (' + str(tiles_country[category]) + ')'
        legend_handles.append(mpatches.Patch(color=colormap(i*120), label=label_str))
    plt.legend(handles=legend_handles, loc='best', bbox_to_anchor=(0, 0.64))
    plt.savefig("8_waffle_.png")
    plt.show()
