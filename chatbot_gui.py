# Streamlined version of chatbot_gui.py
import os
import sys
import json
import sqlite3
import traceback
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import tracemalloc
import yfinance as yf
import logging
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import re
import numpy as np
from chatbot_with_financial_aspect import EnhancedCongressionalChatbot
import plotly.graph_objects as go

# Enable tracemalloc for memory tracking
tracemalloc.start()

def setup_logging():
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    os.makedirs('logs', exist_ok=True)
    
    fh = logging.FileHandler('logs/chatbot.log')
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

def exit_application():
    """Safely exit the application with cleanup."""
    try:
        # Cleanup chatbot if initialized
        if 'chatbot' in st.session_state:
            asyncio.run(st.session_state.chatbot.cleanup())

        # Clear cache and session state
        st.cache_data.clear()
        
        # Close SQLite connections (if applicable)
        if 'db_connection' in st.session_state:
            st.session_state.db_connection.close()
            del st.session_state.db_connection  # Remove from session state
        
        # Close logging handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Display shutdown message
        st.success("Goodbye! Application shutting down...")
        os._exit(0)

    except Exception as e:
        st.error(f"Error during shutdown: {str(e)}")
        st.code(traceback.format_exc())
        os._exit(1)

async def initialize_chatbot():
    """Initialize chatbot asynchronously."""
    try:
        config_path = Path("config/config.json")
        if not config_path.exists():
            raise FileNotFoundError("Configuration file not found!")

        with config_path.open() as f:
            config = json.load(f)

        chatbot = EnhancedCongressionalChatbot(
            db_path=str(Path(config["db_path"]) / "embeddings.sqlite3"),
            env_path=config["env_path"]
        )
        await chatbot.initialize_database()
        return chatbot
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        raise

def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = ""
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = asyncio.run(initialize_chatbot())
            st.session_state.db_stats = asyncio.run(st.session_state.chatbot.get_database_stats())
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.stop()

def format_reference(source):
    """Format a single reference source."""
    return f"""**Congressional Hearing {source['hearing_identifier']}**  
    Congress: {source['congress_number']}  
    Type: {source['hearing_type'].title()}  
    """

def clear_conversation():
    """Clear the conversation history."""
    st.session_state.messages = []
    st.session_state.conversation_context = ""
    if 'chatbot' in st.session_state:
        asyncio.run(st.session_state.chatbot.clear_chat_history())

async def get_chat_response(query: str):
    """Get response from chatbot asynchronously."""
    try:
        return await st.session_state.chatbot.get_enhanced_response(query)
    except Exception as e:
        logger.error(f"Error getting response: {str(e)}")
        return "Error processing your request.", [], {}

async def get_market_data(symbol, chatbot):
    """Get market data for a symbol using the chatbot's market analyzer."""
    try:
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        market_data = await chatbot.market_analyzer.get_market_data(symbol, start_date, end_date)
        predictions = await chatbot.market_analyzer.predict_future_prices(market_data)
        
        # Format data for frontend
        historical_prices = [{
            'date': date.strftime('%Y-%m-%d'),
            'price': price
        } for date, price in zip(market_data.index, market_data['Close'])]
        
        return {
            'currentPrice': market_data['Close'].iloc[-1],
            'priceChange': ((market_data['Close'].iloc[-1] - market_data['Close'].iloc[0]) / market_data['Close'].iloc[0] * 100).round(2),
            'volume': market_data['Volume'].iloc[-1],
            'volatility': market_data['Close'].std(),
            'historicalPrices': historical_prices,
            'predictions': predictions,
            'marketInsights': [
                f"{'Positive' if market_data['Close'].iloc[-1] > market_data['Close'].iloc[0] else 'Negative'} trend over the past year",
                f"Average daily volume: {market_data['Volume'].mean():,.0f}",
                f"Predicted trend: {'Upward' if predictions['Prediction'].iloc[-1] > market_data['Close'].iloc[-1] else 'Downward'}"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        raise

def extract_stock_recommendations(content):
    recommendations = []
    sectors = {
        'real estate': 'IYR',
        'technology': 'XLK', 
        'environmental': 'XLE',
        'clean energy': 'ICLN',
        'renewable energy': 'TAN',
        'defense': 'ITA',
        'aerospace': 'ITA',
        'financial': 'XLF',
        'banking': 'KBE',
        'healthcare': 'XLV',
        'biotech': 'IBB',
        'pharmaceuticals': 'PJP',
        'construction': 'XHB',
        'homebuilders': 'ITB',
        'utilities': 'XLU',
        'consumer': 'XLY',
        'retail': 'XRT',
        'industrial': 'XLI',
        'materials': 'XLB',
        'cybersecurity': 'HACK',
        'semiconductors': 'SOXX',
        'artificial intelligence': 'BOTZ',
        'electric vehicles': 'DRIV',
        'infrastructure': 'PAVE',
        'regulatory compliance': 'XLF',
        'political activities': 'XLF'
    }

    lines = content.split('\n')
    for line in lines:
        # Check for specific ticker mentions
        if "(" in line and ")" in line:
            ticker_match = re.search(r'\(([A-Z]{1,5})\)', line)
            if ticker_match:
                ticker = ticker_match.group(1)
                recommendations.append(('stock', ticker))
        
        # Check for sector mentions
        for sector, etf in sectors.items():
            if sector.lower() in line.lower():
                recommendations.append(('sector', etf))
    
    return list(dict.fromkeys(recommendations))

def display_conversation():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message["role"] == "assistant":
                recommendations = extract_stock_recommendations(message["content"])
                if recommendations:
                    display_financial_data(recommendations)

def get_stock_name(ticker):
    """Get company name for a stock ticker."""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        return info.get('longName', ticker)
    except:
        return ticker

def display_financial_data(recommendations):
    """Display financial data for each recommendation."""
    sector_names = {
        'IYR': 'Real Estate',
        'XLK': 'Technology',
        'XLE': 'Energy',
        'ICLN': 'Clean Energy',
        'TAN': 'Renewable Energy',
        'ITA': 'Aerospace & Defense',
        'XLF': 'Financial',
        'KBE': 'Banking',
        'XLV': 'Healthcare',
        'IBB': 'Biotech',
        'PJP': 'Pharmaceuticals',
        'XHB': 'Homebuilders',
        'ITB': 'Construction',
        'XLU': 'Utilities',
        'XLY': 'Consumer',
        'XRT': 'Retail',
        'XLI': 'Industrial',
        'XLB': 'Materials',
        'HACK': 'Cybersecurity',
        'SOXX': 'Semiconductors',
        'BOTZ': 'Artificial Intelligence',
        'DRIV': 'Electric Vehicles',
        'PAVE': 'Infrastructure'
    }

    timestamp = datetime.now().strftime('%H%M%S')

    for i, (rec_type, symbol) in enumerate(recommendations):
        try:
            data = yf.download(symbol, start=datetime.now() - timedelta(days=365), end=datetime.now(), progress=False)
            
            if not data.empty:
                ticker_obj = yf.Ticker(symbol)
                title = (f"Stock Analysis: {ticker_obj.info.get('longName', symbol)} ({symbol})" 
                        if rec_type == "stock" 
                        else f"Sector Analysis: {sector_names.get(symbol, symbol)} ({symbol})")

                with st.expander(f"📈 {title}"):
                    col1, col2, col3 = st.columns(3)
                    current_price = data['Close'].iloc[-1]
                    price_change = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100)
                    
                    col1.metric("Price", f"${current_price:.2f}", f"{price_change:+.2f}%")
                    col2.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                    col3.metric("Volatility", f"{data['Close'].pct_change().std() * np.sqrt(252):.1%}")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["Price History", "Peer Comparison", "Key Statistics", "Predictions"])
                    
                    with tab1:
                        # Price History chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            name='Price',
                            line=dict(color='blue')
                        ))
                        fig.update_layout(
                            title="Price History",
                            xaxis_title="Date",
                            yaxis_title="Price ($)"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"price_hist_{symbol}_{i}_{timestamp}")

                        historical_volume = st.checkbox("Show Volume", key=f"vol_check_{symbol}_{i}_{timestamp}")
                        if historical_volume:
                            fig_vol = go.Figure()
                            fig_vol.add_trace(go.Bar(
                                x=data.index,
                                y=data['Volume'],
                                name='Volume'
                            ))
                            fig_vol.update_layout(
                                title="Volume History",
                                xaxis_title="Date",
                                yaxis_title="Volume"
                            )
                            st.plotly_chart(fig_vol, use_container_width=True, key=f"vol_chart_{symbol}_{i}_{timestamp}")
                    
                    with tab2:
                        if hasattr(st.session_state, 'chatbot') and hasattr(st.session_state.chatbot, 'financial_analyzer'):
                            with st.spinner("Loading peer comparison..."):
                                try:
                                    analysis = asyncio.run(st.session_state.chatbot.financial_analyzer.analyze_recommendation(symbol, datetime.now()))
                                    
                                    if analysis and 'peer_comparison' in analysis and analysis['peer_comparison']:
                                        peer_df = pd.DataFrame.from_dict(analysis['peer_comparison'], orient='index')
                                        if not peer_df.empty:
                                            st.subheader("Peer Performance Comparison")
                                            
                                            styled_df = peer_df.style.format({
                                                'return': '{:,.2f}%',
                                                'volatility': '{:,.2f}%',
                                                'sharpe': '{:.2f}',
                                                'volume': '{:,.0f}'
                                            }).background_gradient(cmap='RdYlGn', subset=['return', 'sharpe'])\
                                            .background_gradient(cmap='RdYlGn_r', subset=['volatility'])
                                            
                                            st.dataframe(styled_df, use_container_width=True)
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                selected_return = peer_df.loc[f"{symbol} (Selected)", 'return']
                                                avg_return = peer_df['return'].mean()
                                                st.metric("Return vs Peers", f"{selected_return:.2f}%", 
                                                         f"{selected_return - avg_return:+.2f}% vs avg")
                                                
                                                selected_sharpe = peer_df.loc[f"{symbol} (Selected)", 'sharpe']
                                                avg_sharpe = peer_df['sharpe'].mean()
                                                st.metric("Sharpe Ratio vs Peers", f"{selected_sharpe:.2f}",
                                                         f"{selected_sharpe - avg_sharpe:+.2f} vs avg")
                                                
                                            with col2:
                                                selected_vol = peer_df.loc[f"{symbol} (Selected)", 'volatility']
                                                avg_vol = peer_df['volatility'].mean()
                                                st.metric("Volatility vs Peers", f"{selected_vol:.2f}%",
                                                         f"{selected_vol - avg_vol:+.2f}% vs avg",
                                                         delta_color="inverse")
                                                
                                                rank_return = peer_df['return'].rank(ascending=False)[f"{symbol} (Selected)"]
                                                st.metric("Return Ranking", f"#{int(rank_return)} of {len(peer_df)}",
                                                         "Based on return %")
                                        else:
                                            st.info("No peer comparison data available")
                                    else:
                                        st.info("Could not retrieve peer comparison data")
                                except Exception as e:
                                    st.error(f"Error in peer comparison: {str(e)}")
                                    
                    with tab3:
                        stats_col1, stats_col2 = st.columns(2)
                        with stats_col1:
                            yearly_high = data['High'].max()
                            yearly_low = data['Low'].min()
                            avg_volume = data['Volume'].mean()
                            
                            st.metric(
                                label="52-Week High",
                                value=f"${yearly_high:.2f}",
                                delta=f"{((current_price - yearly_high) / yearly_high * 100):.1f}% from high"
                            )
                            st.metric(
                                label="52-Week Low",
                                value=f"${yearly_low:.2f}",
                                delta=f"{((current_price - yearly_low) / yearly_low * 100):+.1f}% from low"
                            )
                        
                        with stats_col2:
                            st.metric(
                                label="Average Daily Volume",
                                value=f"{avg_volume:,.0f}",
                                delta=f"{((data['Volume'].iloc[-1] - avg_volume) / avg_volume * 100):+.1f}% vs avg"
                            )
                            st.metric(
                                label="Volatility (Annualized)",
                                value=f"{data['Close'].pct_change().std() * np.sqrt(252):.1%}"
                            )
                            
                    with tab4:
                        st.subheader("Price Predictions")
                        with st.spinner("Generating predictions..."):
                            try:
                                analysis = asyncio.run(
                                    st.session_state.chatbot.financial_analyzer.predict_returns(
                                        symbol, data, days=30
                                    )
                                )
                                
                                if analysis and 'predictions' in analysis:
                                    pred_df = analysis['predictions']
                                    
                                    historical = data['Close'].tail(30)
                                    future = pred_df['Prediction']
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=historical.index, 
                                        y=historical,
                                        name='Historical',
                                        line=dict(color='blue')
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=future.index,
                                        y=future,
                                        name='Predicted',
                                        line=dict(color='red', dash='dash')
                                    ))

                                    if 'Lower_CI' in pred_df.columns and 'Upper_CI' in pred_df.columns:
                                        fig.add_trace(go.Scatter(
                                            x=future.index,
                                            y=pred_df['Upper_CI'],
                                            line=dict(color='rgba(0,100,255,0.2)'),
                                            name='Upper CI',
                                            showlegend=False
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=future.index,
                                            y=pred_df['Lower_CI'],
                                            fill='tonexty',
                                            line=dict(color='rgba(0,100,255,0.2)'),
                                            name='Lower CI',
                                            showlegend=False
                                        ))

                                    fig.update_layout(
                                        title=f"30-Day Price Prediction for {symbol}",
                                        xaxis_title="Date",
                                        yaxis_title="Price ($)",
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True, key=f"pred_plot_{symbol}_{i}_{timestamp}")
                                    
                                    col1, col2 = st.columns(2)
                                    pred_return = ((future.iloc[-1] - current_price) / current_price * 100)
                                    with col1:
                                        st.metric(
                                            "Predicted Return (30d)", 
                                            f"{pred_return:+.2f}%",
                                            help="Predicted return over next 30 days"
                                        )
                                    with col2:
                                        st.metric(
                                            "Prediction Confidence",
                                            f"{analysis.get('confidence', 0):.1%}",
                                            help="Model's confidence in prediction"
                                        )
                                        
                            except Exception as e:
                                st.error(f"Error generating predictions: {str(e)}")

        except Exception as e:
            st.error(f"Error loading data for {symbol}: {str(e)}")
            
def main():
    """Main application function."""
    st.set_page_config(
        page_title="Congressional Hearing Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("📊 Dashboard")

        if st.button("Exit Application", type="primary"):
            exit_application()

        st.divider()

        if st.button("Clear Conversation"):
            clear_conversation()

        # Database Stats
        st.subheader("Database Statistics")
        if 'db_stats' in st.session_state:
            stats = st.session_state.db_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Embedded Chunks", stats.get('document_count', 'N/A'))
            with col2:
                st.metric("Database Size (MB)", f"{stats.get('database_size', 0):.2f}")
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Documents", stats.get('num_collections', 'N/A'))
            with col4:
                st.metric("Distinct Chunks", stats.get('num_segments', 'N/A'))
        else:
            st.warning("Database statistics not available")

    # Main content
    st.title("Congressional Hearing Analysis System")
    display_conversation()

    if user_query := st.chat_input("Ask a question about congressional hearings"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    conversation_context = ""
                    if len(st.session_state.messages) > 1:
                        conversation_context = "\nPrevious conversation:\n"
                        for msg in st.session_state.messages[:-1]:
                            conversation_context += f"{msg['role'].title()}: {msg['content']}\n"
                    
                    conversation_context += f"\nCurrent question: {user_query}"

                    response, sources, metadata = asyncio.run(
                        st.session_state.chatbot.get_enhanced_response(conversation_context)
                    )

                    st.markdown(response)
                    
                    # Store complete message with metadata
                    message_data = {
                        "role": "assistant",
                        "content": response,
                        "metadata": {
                            "sources": sources,
                            "analysis_data": metadata
                        }
                    }
                    st.session_state.messages.append(message_data)
                    
                    # Display sources if available
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.write(f"{i}. {source.get('hearing_identifier', 'Unknown Hearing')}")
                                                    
                    # Extract and display financial data
                    recommendations = extract_stock_recommendations(response)
                    if recommendations:
                        display_financial_data(recommendations)
                                
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())

# Register cleanup
import atexit
atexit.register(lambda: asyncio.run(st.session_state.chatbot.cleanup()) if 'chatbot' in st.session_state else None)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error in main: {str(e)}")
        st.code(traceback.format_exc())