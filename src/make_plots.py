# Scatter Plot
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime

def make_plots(results_df):
    now = datetime.datetime.now()
    date_stamp = now.strftime("%y%m%d")

    # Create the "./plots" folder with date stamp if it doesn't exist
    plots_folder = f"./plots_{date_stamp}"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    unique_group_gene_values = results_df['Gene_Group'].unique()

    for group_gene in unique_group_gene_values:    
        filtered_df = results_df[(results_df['Gene_Group'] == group_gene) & (results_df['Status'] == 'Active') & (results_df['MSAlimNum'] > 5)]

        # Create a scatter plot with Min_B_Factor_Act_Loop on the x-axis and RMSD_After_Alignment on the y-axis, colored by Conformation
        plt.figure(figsize=(10, 8))

        # Define a custom color style
        n_colors = len(filtered_df["Conformation"].unique())+1  # Change this to the number of unique conformations you need
        palette = sns.color_palette("husl", n_colors)
        
        # Define a custom palette style without green
        green_index = None
        for idx, color in enumerate(palette):
            if 0.2 < color[1] < 0.8 and color[0] < 0.4:  # Adjust this condition to identify the green color in the palette
                green_index = idx
                break

        if green_index is not None:
            del palette[green_index]

        # Define a custom marker style dictionary for the conformations
        conformation_marker_styles = {}  # Initialize an empty dictionary
        unique_conformations = results_df["Conformation"].unique()
        for conf in unique_conformations:
            conformation_marker_styles[conf] = "o"  # Assign a dot marker to each conformatio

        sns.scatterplot(data=filtered_df, x="Min_PLDDT_Act_Loop", y="RMSD_After_Alignment_actloop40N", 
                        hue="Conformation", style="Conformation", markers=conformation_marker_styles, palette=palette)

        # Set plot labels
        plt.xlabel("Minimum PLDDTs in Activation Loop")
        plt.ylabel("RMSD After Alignment")
        plt.title(f"{group_gene}")

        # Save the plot to the "./plots" folder with date stamp
        plot_filename = f"{os.path.basename(group_gene)}_scatter_plot.png"
        plt.savefig(os.path.join(plots_folder, plot_filename), dpi=300)

        # Show the plot
        plt.show()
