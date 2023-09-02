
# Customer Segmentation

## Objective

The goal of this project is to leverage Unsupervised Learning to create a model capable of clustering shopping mall customers effectively.

## Background Information

Efficient customer segmentation is vital for shopping malls with limited resources. By targeting the most profitable customer segments, malls can optimize their marketing efforts.

## The Dataset

| Feature          | Description                               |
|------------------|-------------------------------------------|
| CustomerID       | Unique customer identifier ranging from 1 to 200 |
| Gender           | Customer gender                           |
| Age              | Customer age                              |
| Income (k$)      | Customer income in thousands of dollars  |
| How Much They Spend | Amount spent by the customer           |

Total Rows: 200

All columns serve as features for this unsupervised task.

## Libraries

- **Pandas and NumPy:** Used for data manipulation and analysis.
- **Seaborn and Matplotlib:** Used for data visualization.
- **StandardScaler (from sklearn.preprocessing):** Applied to columns with different scales - Spending, Income, Age.
- **LabelEncoding (from sklearn.preprocessing):** Applied to the Gender column.
- **t-SNE (t-distributed Stochastic Neighbor Embedding) (from sklearn.manifold):** Effective for non-linear data, used for dimension reduction.

## Preprocessing

Based on Exploratory Data Analysis (EDA), the following data transformations were applied:

- Standard scaling for columns with different scales: Spending, Income, Age.
- Label encoding for the Gender column.

## Clustering Investigation

To assess the effectiveness of clustering, the following aspects were explored:

- **Clustering Tendency**: Used Hopkins Statistics to asses.
- **Dimension Reduction**: Compare PCA vs TSNE and went with TSNE
- **Gender Column**: Assessing the relevance of the gender column through feature_importance
   
## Model Evaluation

To evaluate clustering, several models were compared:

- K-Means 
- Spectral Clustering 
- Agglomerative Clustering 
- Gaussian Mixture Model 
- Affinity Propagation 

## Metrics

To select K-value as well as compare models, silhoutte score was used. 

## Model Interpretation

The Affinity Propagation model was found to be the most suitable for clustering. A surrogate model was employed to interpret the resulting clusters, yielding the following customer segments:

| Cluster Color   | Sample Count  | Customer Type     | Demographic Info                          | Marketing Strategy                                   |
|-----------------|---------------|-------------------|------------------------------------------|------------------------------------------------------|
| Orange          | 23            | Reckless Spender  | High Spender, Low Income                 | Upsell high-profit items such as luxury goods.      |
| Pink            | 39            | Big Ticket        | High Spenders, High Income               | Focus on retention through memberships and loyalty programs. Highlight high-profit items.   |
| Blue            | 38            | Kids              | Low Spenders, Young                      | Encourage spending on trendy items due to youth and naivety.  |
| Green           | 49            | Elderly           | Moderate Spenders                         | Promote bundle deals on regular grocery items for increased sales.   |
| Light Green     | 22            | Cheapskates       | Low Spenders, Middle Aged, Low Income    | Low marketing potential, prioritize other segments.        |
| Purple          | 30            | Value Hunters     | Low Spenders, Middle Aged, High Income   | Utilize decoy pricing and tactics to boost spending.      |

## References

- *Deore, S. (2020) [Really, what is Hopkins statistic?](https://sushildeore99.medium.com/really-what-is-hopkins-statistic-bad1265df4b) - Medium*
- *Eniyei et al. (2018) [Assessing clustering tendency](https://www.datanovia.com/en/lessons/assessing-clustering-tendency/) - Datanovia*
- *Prathmachowksey (2020) [A python implementation for computing the Hopkins’ statistic (Lawson and Jurs 1990) for measuring clustering tendency of data](https://github.com/prathmachowksey/Hopkins-Statistic-Clustering-Tendency) - GitHub*
- *Saji, B. (2023) [Elbow method for finding the optimal number of clusters in K-means](https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/#:~:text=K%20Means%20Clustering%20Using%20the%20Elbow&amp;text=For%20each%20value%20of%20K,plot%20looks%20like%20an%20Elbow.) - Analytics Vidhya*
- *K-means clustering algorithm - [javatpoint](https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning)*
- *Introduction to K-means clustering - [Pinecone](https://www.pinecone.io/learn/k-means-clustering/)*
- *V, K. (2022) [What, why and how of spectral clustering!](https://www.analyticsvidhya.com/blog/2021/05/what-why-and-how-of-spectral-clustering/) - Analytics Vidhya*
- *Pai, P. (2021) [Hierarchical clustering explained](

https://towardsdatascience.com/hierarchical-clustering-explained-e59b13846da8) - Medium*
- *[Plot hierarchical clustering dendrogram](https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html) - scikit-learn*
- *Gajjar, K. (2020) [Cluster analysis with DBSCAN: Density-based spatial clustering of applications with noise](https://medium.com/analytics-vidhya/cluster-analysis-with-dbscan-density-based-spatial-clustering-of-applications-with-noise-6ade1ec23555) - Medium*
- *[Gaussian mixture model selection](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html) - scikit-learn*

