import pandas as pd
import matplotlib.pyplot as plt

class HandleResults :

    def get_best(self, df, dataset, metric) :
        "get best performance per oversampler for a specific dataset"
    
        temp_df = df[df['dataset'] == dataset]
    
        return df.loc[temp_df.groupby('oversampler')[metric].idxmax()].sort_values(metric, ascending=False)

    def select_best(self, df, metric) :
        "get best performance per oversampler for all datasets"
        
        temp = []
        
        for dataset in df['dataset'].unique() :
            temp.append(self.get_best(df, dataset, metric))
            
        return pd.concat(temp)

    def ranking_oversampler(self, df, metric, return_all = False) :
        
        df2 = df.groupby('oversampler').mean().sort_values(metric, ascending=False)
        df3 = df2[[metric, f'{metric}_std']]
        
        if return_all :
            return df2
        else :
            return df3
        
    def final_ranking(self, df, metric) :
        
        df2 = self.select_best(df, metric)
        
        return self.ranking_oversampler(df2,metric)

    def plot_bar_metric(self, df, metric) :

        df2 = self.final_ranking(df, metric)
        fig, ax = plt.subplots(figsize=(12,7))
        ax.set_title(metric)
        mean = df2[metric]
        std = df2[f'{metric}_std']
        mean.plot(kind='bar', yerr=std, width=0.8)
        plt.xticks(rotation=45, ha='right')
        plt.show()

        return fig, ax

    def first_places(self, df, metric) :

        best = self.select_best(df, metric)
        df2 = best.loc[best.groupby('dataset')[metric].idxmax()]

        return df2['oversampler'].value_counts().to_frame(metric)
