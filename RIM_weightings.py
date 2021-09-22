import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly
import plotly.graph_objects as go

# Data preparation


def assign_age_group(age):
    if age < 20:
        return 0
    elif age > 50:
        return 2
    else:
        return 1


def prepare_data():

    df_panelists = pd.read_excel('praca_domowa.xlsx', index_col=0, skiprows=9)
    df_panelists.columns = ['Gender', 'Age']  # female 0, male 1
    df_targets = pd.read_excel('praca_domowa.xlsx', nrows=5, header=None, skiprows=2, usecols=[0, 1])
    df_targets.columns = ['Condition', 'Target_pct']

    n_of_panelists = df_panelists.shape[0]

    df_targets['Target_value'] = n_of_panelists*df_targets['Target_pct']
    df_targets['Factor'] = None

    df_targets.loc[:2, 'Factor'] = 'Gender'
    df_targets.loc[2:, 'Factor'] = 'Age_group'

    df_targets['Condition_value'] = None

    for f in df_targets['Factor'].unique():
        n = df_targets[df_targets['Factor'] == f].shape[0]
        df_targets.loc[df_targets['Factor'] == f, 'Condition_value'] = [x for x in range(n)]

    df_panelists['Age_group'] = df_panelists['Age'].apply(assign_age_group)

    df_panelists['weights'] = 1

    return df_panelists, df_targets

# RIM weightings


def update_weights(panelists, targets, factor):
    
    sum_weights = panelists.groupby(factor).sum()['weights'].values
    targets_values = targets[targets['Factor'] == factor]['Target_value'].values
    multipliers = targets_values/sum_weights

    for unq, mul in zip(panelists[factor].sort_values().unique(), multipliers):
        panelists.loc[panelists[factor] == unq, 'weights'] *= mul
        
    return panelists['weights']


def update_weights_dd(panelists, targets, factor):

    sum_weights = panelists.groupby(factor).sum()['weights'].values
    targets_values = targets[targets['Factor'] == factor]['Target_value'].values

    diffs = (targets_values - sum_weights)/panelists[factor].value_counts().values

    for unq, diff in zip(panelists[factor].sort_values().unique(), diffs):
        panelists.loc[panelists[factor] == unq, 'weights'] += diff

    return panelists['weights']


def calculate_error(panelists, targets, error_type):

    values = []
    target_values = []

    for factor in targets['Factor'].unique():
        values.extend(panelists.groupby(factor).sum()['weights'].values)
        target_values.extend(targets[targets['Factor'] == factor]['Target_value'].values)

    if error_type == 'mae':
        return mean_absolute_error(values, target_values)
    elif error_type == 'mse':
        return mean_squared_error(values, target_values)
    elif error_type == 'rmse':
        return mean_squared_error(values, target_values, squared=False)
    else:
        return None


def rim(panelists, targets, n_iter=10, dd=False):

    errors = []
    errors_types = ['rmse', 'mae']

    for n in range(n_iter):
        for factor in targets['Factor'].unique():
            if dd:
                panelists['weights'] = update_weights_dd(panelists, targets, factor)
            else:
                panelists['weights'] = update_weights(panelists, targets, factor)

        errors.append([calculate_error(panelists, targets, e_type) for e_type in errors_types])

        if calculate_error(panelists, targets, 'rmse') < 1e-8:
            break

    return panelists, pd.DataFrame(errors, columns=errors_types)


def visualise_error(errors, errors_dd):
    fig = go.Figure(go.Scatter(x=errors.index, y=errors['mae'], name='default', mode='markers'))
    fig.add_trace(go.Scatter(x=errors_dd.index, y=errors_dd['mae'], name='DD', mode='markers'))
    fig.update_layout(title='MAE for RIM methods', xaxis_title='Iteration', yaxis_title='MAE')
    fig.show()

    plotly.io.write_image(fig, 'MAE_graph.pdf', format='pdf')


if __name__ == '__main__':
    df_panelists, df_targets = prepare_data()
    df_rim, df_errors = rim(deepcopy(df_panelists), deepcopy(df_targets))
    df_rim_dd, df_errors_dd = rim(deepcopy(df_panelists), deepcopy(df_targets), dd=True)
    visualise_error(df_errors, df_errors_dd)
