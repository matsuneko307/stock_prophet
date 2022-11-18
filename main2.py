# 必要なライブラリをインポート
import pandas as pd
import numpy as np
import datetime as dt

from fbprophet import Prophet
import optuna
import streamlit as st

from pandas_datareader import data





st.title('株価予測アプリ')
st.write('任意の銘柄のコードを入力して、当日から3か月後までの株価の値動きを予想をするアプリです')

code = st.text_input('1銘柄分の証券コードを半角で入力後、Enter を押してください')
start = dt.date(2017,1,1)


if st.button('予測開始') and code:
    
    #銘柄の株価を取得
    df = data.DataReader(f"{code}.JP", 'stooq', start)
    if not list(df.columns):
        st.write("このコードの銘柄はありません。上場している銘柄のコードを入力してください")
    else:
        df = df.reset_index()

        #学習用データの準備
        df2 = df.drop(['High', 'Low', 'Open', 'Volume'], axis=1)
        df2.columns = ['ds', 'y']
        
        train = df2[:-60]
        test = df2[-60:]

        #prophetで学習
        def objective(trial):
            prophet_params = {'changepoint_range':trial.suggest_discrete_uniform('changepoint_range', 0.70, 0.95, 0.01),
                            'n_changepoints': trial.suggest_int('n_changepoints', 20, 40),
                            'changepoint_prior_scale': trial.suggest_discrete_uniform('changepoint_prior_scale', 0.001, 0.5, 0.001),
                            'seasonality_prior_scale': trial.suggest_discrete_uniform('seasonality_prior_scale', 0.01, 25, 0.5),
                            'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
                            'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [True, False]),
                            'daily_seasonality': trial.suggest_categorical('daily_seasonality', [True, False]),
                            }
            m = Prophet(changepoint_range = prophet_params['changepoint_range'],
                        n_changepoints = prophet_params['n_changepoints'],
                        changepoint_prior_scale = prophet_params['changepoint_prior_scale'],
                        seasonality_prior_scale = prophet_params['seasonality_prior_scale'],
                        yearly_seasonality = prophet_params['yearly_seasonality'], 
                        weekly_seasonality = prophet_params['weekly_seasonality'], 
                        daily_seasonality = prophet_params['daily_seasonality'], 
                        holidays = None,
                        growth = 'linear'
                        )
            m.fit(train)
            forecast = m.predict(df2)
            valid_forecast = forecast.tail(len(test))
            
            #MAPE=平均絶対パーセント誤差
            val_mape = np.mean(np.abs((valid_forecast.yhat - test.y) / test.y))*100
        
            return val_mape


        study = optuna.create_study(direction="minimize") 
        study.optimize(objective, n_trials=3)

        #予測と実測との平均絶対誤差をテストデータを使って求める
        #作成したモデルを使って90日分先を予測
        m = Prophet(**study.best_params)
        m.fit(df2)
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)

        #グラフで実測値と予測値を比較するためのデータを用意
        df_plot = forecast[["ds","yhat"]]
        #dfとdf_plotは日付が逆になっているため、dfの行の並びを逆にする
        sort_df = df.sort_index(ascending=False)
        Actual_value = sort_df["Close"].values
        df_acc = pd.DataFrame({'Close': Actual_value})
        df_plot = pd.concat([df_plot, df_acc],axis=1)
        df_plot = df_plot.set_index("ds")
        df_plot = df_plot.set_axis(['予測値', '実測値'], axis=1)

        #streamlit にグラフを表示
        st.subheader('実測株価とモデル予測株価の表示')
        st.line_chart(df_plot)

        #実測と予測の表を表示
        st.write("グラフのデータ内容")
        st.dataframe(df_plot)

        
        

        
        
elif code == False:
    st.title('コードを入力してください')




