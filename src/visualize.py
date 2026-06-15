import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

from .utils import *


def barplot_acc(accuracies, rec='center'):

    df = pd.DataFrame(accuracies, columns=['acc'])
    df['Question Asked'] = list(np.arange(len(accuracies)))
    fig = px.bar(df, x='Question Asked', y='acc')
    fig.write_image(f"acc_{rec}.png")
    fig.update_layout(title=f'HIT@20 {rec}')


def multiple_barplot(set_acc, set_rec, name='ndcg', save_dir=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    path = os.path.join(save_dir, name + '.pdf')

    fig=go.Figure()
    fig.write_image(path)

    df = pd.DataFrame(set_acc[0], columns=[name])

    df['Q&A Round'] = list(np.arange(len(set_acc[0])))
    df['type'] = [set_rec[0]] * len(set_acc[0])
    for i in range(1, len(set_acc)):
        df_ = pd.DataFrame(set_acc[i], columns=[name])
        df_['Q&A Round'] = list(np.arange(len(set_acc[i])))
        df_['type'] = [set_rec[i]] * len(set_acc[i])
        df = pd.concat([df, df_])
    # df.to_csv("name+".csv", index=False)
    fig = px.bar(df, x='Q&A Round', y=name, color='type', barmode='group', log_y=True)
    fig.write_image(path)
    # fig.update_layout(title=f'HIT@20')


def visualize_recommendation(item_embeddings, chebyshev_user, user, inbound_ids, outbound_ids, boundaries, center_ids, new_item=None, iteration=1, rec='center', n_popular=None):
    fig = go.Figure()

    chebyshev_user = np.squeeze(chebyshev_user)[np.newaxis, :]
    user = np.squeeze(user)[np.newaxis, :]
    X = np.concatenate((item_embeddings, chebyshev_user, user), axis=0)
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        components = normalize(components)
    else:
        components = X.copy()
    inbound_items = components[inbound_ids]
    outbound_items = components[outbound_ids]
    centers = components[center_ids]
    chebyshev = components[-2]
    true_user = components[-1]

    fig.add_trace(go.Scatter(
                    mode='markers', x=inbound_items[:, 0], y=inbound_items[:, 1], opacity=0.8,
                    name=f'Item set U',
                    marker=dict(color='lightgreen', size=5),
                    showlegend=True
                ))
    
    fig.add_trace(go.Scatter(
                    mode='markers', x=outbound_items[:, 0], y=outbound_items[:, 1], opacity=0.8,
                    name=f'Item out of bound',
                    marker=dict(color='grey', size=5),
                    showlegend=True
                ))
    
    fig.add_trace(go.Scatter(
                    mode='markers', x=centers[:, 0], y=centers[:, 1], opacity=1,
                    name=f'centers', marker_symbol='star-diamond',
                    marker=dict(color='MediumPurple', size=10),
                    showlegend=True
                ))
    if new_item is not None:
        fig.add_trace(go.Scatter(
                    mode='markers', x=[components[new_item, 0]], y=[components[new_item, 1]], opacity=1,
                    name=f'item k+1', marker_symbol='star-diamond',
                    marker=dict(color='darkkhaki', size=10),
                    showlegend=True
                ))
    fig.add_trace(
        go.Scatter(
            mode='markers', x=[chebyshev[0]], y=[chebyshev[1]], opacity=1,
            marker_symbol='star', name=f'Chebeshev user',
            marker=dict(color='blue', size=15),
            showlegend=True
        ))
    
    fig.add_trace(
        go.Scatter(
            mode='markers', x=[true_user[0]], y=[true_user[1]], opacity=1,
            marker_symbol='star', name=f'true user',
            marker=dict(color='darkred', size=15),
            showlegend=True
        ))

    for boundary in boundaries:
        w, b = boundary
        # print(f'y = {-w[0]/w[1]}x + {-b/w[1]}')
        x = [-1, 1]
        y1 = -w[0]*x[0]/w[1] - b/w[1]
        y2 = -w[0]*x[1]/w[1] - b/w[1]
        # print(y1, y2)
        y = [y1, y2]
        fig.add_trace(
            go.Scatter(x=x, y=y, line_shape='linear', showlegend=False)
            )

    gt = get_k_closest_items(item_embeddings, user, k=20)
    if rec == 'center':
        pred = get_k_closest_items(item_embeddings, chebyshev_user, k=20)
    elif rec == 'ask_popular':
        pred = get_k_closest_items(item_embeddings, chebyshev_user, k=20)
    elif rec == 'rec_popular':
        pred = get_k_popular_items(inbound_ids, outbound_ids, k=20, n_popular=n_popular)

    gt = list(np.squeeze(gt))
    pred = list(np.squeeze(pred))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    if rec == 'center': 
        fig.update_layout(title=f'Similarity of User and Chebyshev {(euclidean_distances(chebyshev_user.reshape(1,-1), user.reshape(1,-1)))[0][0]}\nHIT@20: {accuracy(pred,gt)}', yaxis_range=[0, 1], xaxis_range=[0, 1])
    elif rec == 'ask_popular':
        fig.update_layout(title=f'Similarity of User and Chebyshev {euclidean_distances(chebyshev_user.reshape(1,-1), user.reshape(1,-1))}\nHIT@20: {accuracy(pred,gt)}', yaxis_range=[0, 1], xaxis_range=[0, 1])
    elif rec == 'rec_popular':
        fig.update_layout(title=f'#popular items in set U: {sum(np.array(inbound_ids) < n_popular)} - HIT@20: {accuracy(pred,gt)}', yaxis_range=[0, 1], xaxis_range=[0, 1])
    
    fig.write_image(f'{rec}_recommendation_{iteration}.png')
    


