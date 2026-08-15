import math
import os
from datetime import date
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SYMBOLS={"Apex Cloud (APX)":(82,.0007,.018,11),"Northstar EV (NEV)":(146,.0003,.026,29),"Maple Energy (MPE)":(54,.0005,.014,47),"Harbor Markets ETF (HMX)":(310,.0004,.009,83)}
MODELS={"Random Forest":lambda:RandomForestRegressor(n_estimators=180,max_depth=9,min_samples_leaf=4,random_state=7,n_jobs=-1),"Gradient Boosting":lambda:GradientBoostingRegressor(n_estimators=140,max_depth=3,learning_rate=.035,loss="huber",random_state=7),"Ridge Regression":lambda:make_pipeline(StandardScaler(),Ridge(alpha=8))}
FEATURES=["close","return_1d","sma_9","sma_21","ema_12","momentum_5","volatility_10","volume_ratio","range_from_sma"]

def market_data(label):
    start,drift,volatility,seed=SYMBOLS[label];rng=np.random.default_rng(seed);dates=pd.bdate_range("2021-01-04",date.today());regime=np.where(np.arange(len(dates))%430<280,1,-.45);cycle=np.sin(np.arange(len(dates))/34)*.002;returns=drift*regime+cycle+rng.normal(0,volatility,len(dates));close=start*np.exp(np.cumsum(returns));volume=rng.integers(700_000,9_000_000,len(dates))*(1+np.abs(returns)*8);return pd.DataFrame({"date":dates,"close":close,"volume":volume})

def engineer(df):
    out=df.copy();out["return_1d"]=out.close.pct_change();out["sma_9"]=out.close.rolling(9).mean();out["sma_21"]=out.close.rolling(21).mean();out["ema_12"]=out.close.ewm(span=12).mean();out["momentum_5"]=out.close.pct_change(5);out["volatility_10"]=out.return_1d.rolling(10).std();out["volume_ratio"]=out.volume/out.volume.rolling(20).mean();out["range_from_sma"]=out.close/out.sma_21-1;out["target"]=out.close.shift(-1);return out.dropna().reset_index(drop=True)

def max_drawdown(equity):return float((equity/equity.cummax()-1).min())

def run(symbol,selected_model,capital,test_percent,threshold,fee_bps):
    df=engineer(market_data(symbol));split=int(len(df)*(1-test_percent/100));train,test=df.iloc[:split],df.iloc[split:].copy();rows=[];predictions={};fitted={};baseline=test.close.to_numpy()
    rows.append(["Previous close baseline",mean_absolute_error(test.target,baseline),math.sqrt(mean_squared_error(test.target,baseline)),r2_score(test.target,baseline),np.mean(np.sign(baseline-test.close)==np.sign(test.target-test.close))*100])
    for name,factory in MODELS.items():
        model=factory().fit(train[FEATURES],train.target);prediction=model.predict(test[FEATURES]);predictions[name]=prediction;fitted[name]=model;rows.append([name,mean_absolute_error(test.target,prediction),math.sqrt(mean_squared_error(test.target,prediction)),r2_score(test.target,prediction),np.mean(np.sign(prediction-test.close)==np.sign(test.target-test.close))*100])
    board=pd.DataFrame(rows,columns=["Model","MAE ($)","RMSE ($)","R²","Direction accuracy (%)"]).sort_values("MAE ($)");board.iloc[:,1:]=board.iloc[:,1:].round(3)
    test["prediction"]=predictions[selected_model];predicted_return=test.prediction/test.close-1;actual_return=test.target/test.close-1;test["signal"]=(predicted_return>threshold/100).astype(int);turnover=test.signal.diff().abs().fillna(test.signal);test["strategy_return"]=test.signal*actual_return-turnover*fee_bps/10000;strategy=float(capital)*(1+test.strategy_return).cumprod();hold=float(capital)*(1+actual_return).cumprod();strategy_return=strategy.iloc[-1]/capital-1;hold_return=hold.iloc[-1]/capital-1
    trades=test.loc[turnover.ne(0),["date","close","prediction","signal"]].tail(15).copy();trades["action"]=trades.signal.map({1:"ENTER",0:"EXIT"});trades=trades[["date","action","close","prediction"]];trades.columns=["Date","Action","Close","Predicted next close"];trades.iloc[:,2:]=trades.iloc[:,2:].round(2)
    model_mae=float(board.loc[board.Model==selected_model,"MAE ($)"].iloc[0]);base_mae=float(board.loc[board.Model=="Previous close baseline","MAE ($)"].iloc[0]);verdict="beats" if model_mae<base_mae else "does not beat"
    summary=f"<div class='score-grid'><article><span>Selected model</span><strong>{selected_model}</strong><small>{verdict} the naive baseline</small></article><article><span>Strategy return</span><strong>{strategy_return:+.1%}</strong><small>after {fee_bps:.0f} bps per position change</small></article><article><span>Buy & hold</span><strong>{hold_return:+.1%}</strong><small>same held-out period</small></article><article><span>Max drawdown</span><strong>{max_drawdown(strategy):.1%}</strong><small>strategy peak-to-trough</small></article></div><p class='readout'><b>Evaluation window:</b> {test.date.min():%b %Y}–{test.date.max():%b %Y} · {len(train):,} training rows · {len(test):,} unseen rows. Bundled series are deterministic market simulations, not live securities.</p>"
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);fig.patch.set_facecolor("#080b12")
    for ax in axes.flat:
        ax.set_facecolor("#0e1420");ax.tick_params(colors="#8993a5",labelsize=8);ax.grid(alpha=.1);[spine.set_color("#263044") for spine in ax.spines.values()]
    axes[0,0].plot(test.date,test.target,color="#55e6a5",label="Actual",linewidth=1.5);axes[0,0].plot(test.date,test.prediction,color="#8b7cff",label="Forecast",linewidth=1.1);axes[0,0].set_title("HELD-OUT FORECAST",color="white",loc="left",fontweight="bold");axes[0,0].legend(facecolor="#0e1420",labelcolor="white")
    axes[0,1].plot(test.date,strategy,color="#55e6a5",label="Signal strategy");axes[0,1].plot(test.date,hold,color="#ffbd59",label="Buy & hold");axes[0,1].set_title("EQUITY CURVE",color="white",loc="left",fontweight="bold");axes[0,1].legend(facecolor="#0e1420",labelcolor="white")
    recent=df.tail(180);axes[1,0].plot(recent.date,recent.close,color="white",label="Close");axes[1,0].plot(recent.date,recent.sma_9,color="#55e6a5",label="SMA 9");axes[1,0].plot(recent.date,recent.sma_21,color="#8b7cff",label="SMA 21");axes[1,0].fill_between(recent.date,recent.sma_9,recent.sma_21,where=recent.sma_9>=recent.sma_21,color="#55e6a5",alpha=.08);axes[1,0].set_title("TECHNICAL SIGNALS",color="white",loc="left",fontweight="bold");axes[1,0].legend(facecolor="#0e1420",labelcolor="white")
    model=fitted[selected_model];importance=pd.Series(np.abs(model[-1].coef_) if selected_model=="Ridge Regression" else model.feature_importances_,index=FEATURES);importance.sort_values().tail(8).plot.barh(ax=axes[1,1],color="#8b7cff");axes[1,1].set_title("MODEL DRIVERS",color="white",loc="left",fontweight="bold")
    return summary,fig,board,trades

CSS="""body,.gradio-container{background:#080b12!important;color:#f5f7fb!important}.gradio-container{max-width:1380px!important;padding:28px!important}.hero{padding:34px;border:1px solid #272d3a;border-radius:22px;background:radial-gradient(circle at 90% 10%,#291d4c 0,transparent 35%),#0d111a;margin-bottom:18px}.hero h1{font-size:48px!important;letter-spacing:-2.5px;margin:0!important}.hero p{color:#a8b0c0;font-size:17px;max-width:760px}.kicker{color:#55e6a5!important;text-transform:uppercase;letter-spacing:3px;font-size:11px!important;font-weight:800}.panel{background:#0d111a!important;border:1px solid #272d3a!important;border-radius:18px!important;padding:10px!important}button.primary{background:linear-gradient(90deg,#55e6a5,#86ffd0)!important;color:#05110c!important;border:0!important;font-weight:800!important}.score-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.score-grid article{background:#0d111a;border:1px solid #272d3a;border-radius:15px;padding:18px}.score-grid span,.score-grid small{display:block;color:#8993a5}.score-grid strong{display:block;color:white;font-size:22px;margin:9px 0}.readout{color:#a8b0c0;line-height:1.7;margin:18px 3px}.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.method-step,.contribution{padding:24px;border:1px solid #272d3a;border-radius:16px;background:#0d111a}.method-step span{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;background:#55e6a519;color:#55e6a5;font-weight:900}.method-step h3,.contribution h3{color:white;margin:18px 0 9px;font-size:18px}.method-step p,.contribution p{color:#a8b0c0;line-height:1.7;margin:0}.contribution{margin-top:14px;background:linear-gradient(120deg,#0d111a,#17142a);border-color:#393052}.contribution small{color:#8b7cff;text-transform:uppercase;letter-spacing:2px;font-weight:800}.disclaimer{color:#8993a5;font-size:12px}@media(max-width:800px){.score-grid,.method-grid{grid-template-columns:1fr 1fr}.hero h1{font-size:36px!important}}"""

with gr.Blocks(css=CSS,title="MarketSignal Intelligence Lab") as demo:
    gr.HTML("<section class='hero'><p class='kicker'>AI market research workbench</p><h1>MarketSignal Intelligence Lab</h1><p>Compare forecasting models, inspect the signals behind their decisions, and stress-test a strategy on data the model never saw during training.</p></section>")
    with gr.Tabs():
        with gr.Tab("Forecast & strategy lab"):
            with gr.Row():
                with gr.Column(scale=1,elem_classes="panel"):
                    symbol=gr.Dropdown(list(SYMBOLS),value=list(SYMBOLS)[0],label="Research market");model=gr.Radio(list(MODELS),value="Random Forest",label="Forecasting model");capital=gr.Number(value=10000,minimum=100,label="Starting capital ($)");test=gr.Slider(15,40,value=25,step=5,label="Unseen test window (%)");threshold=gr.Slider(0,2,value=.25,step=.05,label="Minimum forecast edge (%)");fee=gr.Slider(0,30,value=5,step=1,label="Trading cost (basis points)");button=gr.Button("Run market analysis",variant="primary");gr.Markdown("Educational research demo. Simulated bundled markets; no live prices, brokerage connection, or financial advice.",elem_classes="disclaimer")
                with gr.Column(scale=3):output=gr.HTML();plot=gr.Plot(show_label=False)
            with gr.Row():leaderboard=gr.Dataframe(interactive=False,label="Model leaderboard — lower MAE is better");trades=gr.Dataframe(interactive=False,label="Recent strategy decisions")
        with gr.Tab("How the system works"):
            gr.HTML("""<div class='method-grid'>
<article class='method-step'><span>01</span><h3>Build features safely</h3><p>Returns, <b>SMA9/SMA21</b>, EMA12, momentum, volatility, relative volume, and distance from trend become model inputs without exposing the future target.</p></article>
<article class='method-step'><span>02</span><h3>Respect time order</h3><p>Training uses the earlier series. Evaluation uses the later held-out window—never a random shuffle that could leak future market structure.</p></article>
<article class='method-step'><span>03</span><h3>Make models earn their place</h3><p>Random Forest, Gradient Boosting, and Ridge Regression compete against a previous-close baseline. Weak results remain visible.</p></article>
<article class='method-step'><span>04</span><h3>Test product behavior</h3><p>A configurable forecast threshold becomes an enter/exit signal. The backtest includes costs, buy-and-hold performance, and maximum drawdown.</p></article>
</div><article class='contribution'><small>Original group project · My contribution</small><h3>Indicators, interface, data reliability, and delivery</h3><p>I implemented SMA9/SMA21, improved the graphs and Gradio workflow, helped repair the dataset for multiple stocks, and kept the team moving through delivery. This workbench expands that work into a clearer evaluation product.</p></article>""")
    inputs=[symbol,model,capital,test,threshold,fee];outputs=[output,plot,leaderboard,trades];button.click(run,inputs,outputs);demo.load(run,inputs,outputs)
if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))

