# 必要なライブラリをインポート
import pandas as pd
import numpy as np
import datetime as dt

import plotly.graph_objects as go
from prophet import Prophet
import optuna
import streamlit as st

from pandas_datareader import data




st.set_page_config(layout="wide")
st.title('簡単 "4STEP" でできる株価予測!')
st.write('東京証券取引所に上場している銘柄の将来の株価を少しの入力だけで予測できるアプリです。')
st.write('当アプリの予測には Meta社（旧Facebook社)によって開発された時系列解析用のライブラリ "Prophet" を使用しています。')

st.markdown('# Markdown documents')
st.write("------------------------------------------------------------------------------------------------------------------------------------")
if st.checkbox('<使い方>'):
    st.write("左側を操作します。")
    st.write("step1.に証券コードを半角で入力し、Enter を押してください。予測可能な銘柄の場合、次のステップが表示されます。")
    st.write("step2.で予測を開始する日付を選択します。")
    st.write("step3.で予測開始から何日後までの予測をしたいのか、スライドバーで調整します。")
    st.write("step1,2,3 の全てを設定し終わったら予測開始ボタンを押してグラフが表示されるまで待ちます。グラフが表示されるまでに5分ほどかかります。")

    

#コードを入力後、株価を取得し学習用データを作成する
st.sidebar.write("証券コード入力")
code = st.sidebar.text_input('1銘柄分の証券コードを半角英数字で入力後、Enter を押してください')
if code:
    df_sourse = data.DataReader(f"{code}.JP", 'stooq')
    
    if not list(df_sourse.columns):
        st.sidebar.write("この証券コードの銘柄はありません。上場している銘柄の証券コードを入力してください")
    elif len(df_sourse) < 200:
        st.sidebar.write("この証券コードの銘柄は上場後、200営業日が経過していないので予測できません。")
    else:
        df_sourse = df_sourse.sort_values('Date').reset_index()

        st.sidebar.write("予測開始日")
        start_date = st.sidebar.selectbox('予測開始日を選択してください', list(df_sourse["Date"][200:]))

        st.sidebar.write("予測期間")
        end_date = st.sidebar.slider('予測開始日から何日後まで予測するか、指定してください', 1, 365, 90, 1)
        st.sidebar.write(f"{start_date.date()} ~ {start_date.date() + dt.timedelta(days=end_date)}の期間を予測します")
        
        #学習用データの準備
        if st.sidebar.button('予測開始') and code :
            forecast_start_index = df_sourse.query('Date == @start_date').index[0]
            df = df_sourse[:forecast_start_index]
            df = df.drop(['High', 'Low', 'Open', 'Volume'], axis=1)
            df.columns = ['ds', 'y']
            len_df_train = int(len(df)*0.7)
            train = df[:len_df_train]
            test = df[len_df_train:]
            
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
                forecast = m.predict(df)
                valid_forecast = forecast.tail(len(test))
                
                #MAPE=平均絶対パーセント誤差
                val_mape = np.mean(np.abs((valid_forecast.yhat - test.y) / test.y))*100
            
                return val_mape


            study = optuna.create_study(direction="minimize") 
            study.optimize(objective, n_trials=40)
            
            #予測と実測との平均絶対誤差をテストデータを使って求める
            m = Prophet(**study.best_params)
            m.fit(df)
            future = m.make_future_dataframe(periods=end_date+1)
            forecast = m.predict(future)

            #グラフで実測値と予測値を比較するためのデータを用意
            df_plot = forecast[["ds","yhat"]]
            #予測開始日までの実測値を用意
            df_acc = df_sourse["Close"]
            #実測値と予測値を並べて比較するために結合
            df_plot = pd.concat([df_plot, df_acc],axis=1)
            df_plot = df_plot.set_axis(['Date','予測値', '実測値'], axis=1)
            df_plot["Date"] = pd.to_datetime(df_plot["Date"]).dt.strftime("%Y-%m-%d")
            df_plot["誤差(%)"] = ((df_plot["実測値"][:len(df_acc)] / df_plot["予測値"][:len(df_acc)]) -1) * 100

            df_plot = df_plot.round(1)

            #streamlit にグラフを表示
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot['Date'][:forecast_start_index+1],
                                    y=df_plot['実測値'][:forecast_start_index+1],
                                    mode='lines',
                                    name='実測値',
                                    ),
                        )

            fig.add_trace(go.Scatter(x=df_plot['Date'][forecast_start_index:],
                                    y=df_plot['実測値'][forecast_start_index:],
                                    mode='lines',
                                    name='実測値(予)',
                                    ),
                        )

            fig.add_trace(go.Scatter(x=df_plot['Date'],
                                    y=df_plot['予測値'],
                                    mode='lines',
                                    name='予測値',
                                    ),
                        )
            fig.update_layout(hovermode='x unified')
            
            st.plotly_chart(fig)
            

            

            #実測と予測の表を表示
            st.sidebar.write("グラフのデータ内容")
            st.sidebar.dataframe(df_plot)
     
elif code == False:
    st.title('コードを入力してください')

st.write("------------------------------------------------------------------------------------------------------------------------------------")
st.write("※注意")
st.write('当アプリが行うのは予測であり、将来の株価を確約するものではありません。')
st.write("上場後、200営業日が経過していない銘柄については予測できないようになっております。")
st.write("------------------------------------------------------------------------------------------------------------------------------------")
st.write("※免責事項")
st.write("当アプリの内容およびご利用者様が本サイトを通じて得る情報等について、その正確性、完全性、有用性、最新性、適切性、確実性、動作性等、その内容について何ら法的保証をするものではありません")
st.write("当アプリに掲載されている情報を利用することで発生した紛争や損害に対し、当アプリ制作者は責任を負わないものとします。")




