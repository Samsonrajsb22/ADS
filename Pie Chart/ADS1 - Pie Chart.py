# -*- coding: utf-8 -*-
"""
Spyder Editor

Generated Pie Chart for the Visulisation Assignment
Author: Samson Raj Babu Raj

"""
# importing packages
import pandas as pd
import matplotlib.pyplot as plt

# defining a function


def Top_N_Performers(df_songs_dura, n):
    """
    This function calculates the Top N Performers from the input DataFrame
    Parameters :
        df_songs_dura - DataFrame that contains song details
        N - Integer for Top no.of.Performers
    """
    df_songs_art = df_songs_dura.groupby(["artist"]).count()
    df_song_pie = df_songs_art["song"].sort_values(ascending=False)
    df_song_piee = df_song_pie.head(n)
    df = pd.DataFrame({"artist": df_song_piee.index,
                      "Released": df_song_piee.values})
    return df


# Reading the CSV file
df_songs = pd.read_csv(
    "C:/Users/hp/Desktop/ADS1/Pie Chart/songs_normalize.csv", index_col=0)
df_songs_dura = df_songs.sort_values("duration_ms", ascending=True)

# creating the input
N = int(input("Enter the number of top artist:"))
df = Top_N_Performers(df_songs_dura, N)
total = df["Released"].sum()
total

# plotting the pie graph
plt.pie(df["Released"], labels=df["artist"],
        autopct=lambda p: "{:.0f}%".format(p * total / 100))
title_str = "Songs Released by Top " + str(N) + " Performers"
plt.title(title_str)
plt.axis("Equal")
plt.figure()
plt.show()
