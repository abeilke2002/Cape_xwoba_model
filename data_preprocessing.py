import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def preprocess(data):
    '''
    Create Total Bases Variable from 2024 TM Data
    '''
    data['bip'] = np.where(data['ExitSpeed'] > 0, 1, np.nan)

    plays = [(data['PlayResult'] == 'Single'),
        (data['PlayResult'] == 'Double'),
        (data['PlayResult'] == 'Triple'),
        (data['PlayResult'] == 'HomeRun')]

    plays_tb = [1, 2, 3, 4]
    data['tb'] = np.select(plays, plays_tb, default=0)

    data_bip = data[~data['bip'].isnull()].dropna(subset=['ExitSpeed','Angle','bip'])
    data_bip = data_bip[~((data_bip['PlayResult'] == "Sacrifice") & (data_bip['ExitSpeed'] < 60))]
    data_bip = data_bip[~((data_bip['PlayResult'] == "Undefined"))]

    return data_bip

def create_histogram(data_bip):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data_bip, x='TotalBases', bins=5)
    plt.xlabel('Total Bases')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Bases in 2024 College Data')
    plt.savefig('cluster/plots/total_bases_distribution.png')
    plt.close()

def create_graphic(data_bip):
    data_bip['TotalBases'] = data_bip['tb'].astype('category')
    samp = data_bip.sample(10000)

    custom = {
        0: '#e41a1c',
        1: '#377eb8',
        2: '#4daf4a',
        3: '#984ea3',
        4: '#ff7f00',
    }

    plt.figure(figsize=(10, 6))

    scatter = sns.scatterplot(
        x='ExitSpeed',
        y='Angle',
        hue='TotalBases',
        palette=custom,
        data=samp,
        alpha=0.7
    )

    plt.xlabel('Exit Velocity (mph)')
    plt.ylabel('Launch Angle (degrees)')
    plt.title('Relationship between Exit Velocity, Launch Angle, and Total Bases')
    plt.xlim(0, 120)
    plt.ylim(-100, 100)

    # Show legend
    plt.legend(title='Total Bases')

    # Save the plot
    plt.savefig('cluster/plots/exit_velocity_launch_angle_total_bases.png')
    plt.close()

    create_histogram(data_bip)

def save_train_test_split(data_bip):
    cols_df = data_bip[['ExitSpeed', 'Angle', 'tb']]
    train_df, test_df = train_test_split(cols_df, test_size=0.20, random_state=42)
    train_df.to_csv('cluster/data/train_data.csv', index=False)
    test_df.to_csv('cluster/data/test_data.csv', index=False)


def main():
    file = 'cape_data.csv'
    print('Loading Data:')
    data = pd.read_csv(f"./cluster/data/{file}", low_memory=False)

    data_bip = preprocess(data)
    create_graphic(data_bip)
    save_train_test_split(data_bip)
    print(data_bip['tb'].unique())

if __name__ == "__main__":
    main()


