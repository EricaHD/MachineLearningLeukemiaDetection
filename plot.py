import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
sns.set_style("whitegrid")
sns.set_context("poster")


def show_data():
    """
    Plots features pairwise, of the following set:
    - average allelic fraction
    - hematocrit
    - platelet
    - white blood cell count
    - hemoglobin
    - age
    """

    # Preprocessing
    aml_data = preprocessing.load_csv()
    preprocessing.fill_missing_values(aml_data)
    preprocessing.add_total_genes(aml_data)

    # Delete gene columns
    for column in aml_data.columns:
        if 'Gene.' in column:
            del aml_data[column]

    # Plot pairwise
    sns.set(style='whitegrid')
    cols = ['caseflag', 'Total.Genes', 'Age', 'WBC', 'PLATELET', 'HEMOGLBN', 'HEMATOCR']
    sns.pairplot(aml_data[cols],
                 hue='caseflag',  # different caseflags have different colors
                 markers=['.', r'$+$'],  # markers
                 plot_kws={"s": 250},  # marker size (100 default)
                 size=5.0)  # size of each subplot
    plt.show()
