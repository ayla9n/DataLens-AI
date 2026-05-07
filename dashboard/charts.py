import plotly.express as px
import pandas as pd
import streamlit as st

# Global color config 
COLOR_PALETTE = [
    "#B3CDE3",  
    "#6497B1",  
    "#005B96",  
    "#AED6F1",  
    "#5DADE2",  
    "#2E86C1",  
    "#A9CCE3",  
    "#1A5276",  
]
SINGLE_COLOR = "#5DADE2"   
HEATMAP_SCALE = "Blues"    



def create_chart(df, chart):
    '''
    Creates the correct chart based on AI recommendation type
    '''
    chart_type = chart.get("type")
    title = chart.get("title", "")
    x = chart.get("x")
    y = chart.get("y")
    reason = chart.get("reason", "")

    if x and x not in df.columns:
        st.warning(f"Skipping '{title}' — column '{x}' not found.")
        return
    if y and y not in df.columns:
        st.warning(f"Skipping '{title}' — column '{y}' not found.")
        return

    try:
        match chart_type:
            case "line_chart":
                fig = create_line_chart(df, x, y, title)
            case "bar_chart":
                fig = create_bar_chart(df, x, y, title)
            case "histogram":
                fig = create_histogram(df, x, title)
            case "scatter_plot":
                fig = create_scatter_plot(df, x, y, title)
            case "pie_chart":
                fig = create_pie_chart(df, x, y, title)
            case "heatmap":
                fig = create_heatmap(df, title)
            case _:
                st.warning(f"Unknown chart type: {chart_type}")
                return

        if fig:
            st.plotly_chart(fig, width="stretch")
            st.caption(f"💡 {reason}")

            # Compute accurate takeaways from actual data
            takeaways = get_chart_takeaways(df, chart)
            for point in takeaways:
                st.caption(f" • {point}")

    except Exception as e:
        st.warning(f"Could not create chart '{title}': {e}")


def create_line_chart(df, x, y, title):
    '''
    creates and returnd line chart
    '''
    df_sorted = df.sort_values(by=x)
    fig = px.line(
        df_sorted, x=x, y=y,
        title=title,
        markers=True,
        color_discrete_sequence=[SINGLE_COLOR]
    )
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig


def create_bar_chart(df, x, y, title):
    '''
    creates and returnd bar chart
    '''
    df_grouped = df.groupby(x)[y].sum().reset_index()
    df_grouped = df_grouped.sort_values(by=y, ascending=False)

    fig = px.bar(
        df_grouped, x=x, y=y,
        title=title,
        color=x,
        color_discrete_sequence=COLOR_PALETTE
    )
    fig.update_layout(xaxis_title=x, yaxis_title=y, showlegend=False)
    return fig


def create_histogram(df, x, title):
    '''
    creates and returns histogram
    '''
    fig = px.histogram(
        df, x=x,
        title=title,
        nbins=30,
        color_discrete_sequence=[SINGLE_COLOR]
    )
    fig.update_layout(xaxis_title=x, yaxis_title="Count")
    return fig


def create_scatter_plot(df, x, y, title):
    '''
    creates and returns scatterplot
    '''
    fig = px.scatter(
        df, x=x, y=y,
        title=title,
        opacity=0.6,
        color_discrete_sequence=[SINGLE_COLOR]
    )
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig


def create_pie_chart(df, names, values, title):
    '''
    creates and returns pie chart
    '''
    df_grouped = df.groupby(names)[values].sum().reset_index()
    fig = px.pie(
        df_grouped,
        names=names,
        values=values,
        title=title,
        hole=0.3,
        color_discrete_sequence=COLOR_PALETTE
    )
    return fig


def create_heatmap(df, title):
    '''
    creates and returns heatmap 
    '''
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns to generate a heatmap.")
        return None

    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr,
        title=title,
        color_continuous_scale=HEATMAP_SCALE,
        zmin=-1,
        zmax=1,
        text_auto=True
    )
    return fig


def create_all_charts(df, chart_recommendations):
    '''
    creates all AI recoommended charts
    '''
    if not chart_recommendations:
        st.info("No chart recommendations available.")
        return

    charts = chart_recommendations.get("charts", [])

    if not charts:
        st.info("No charts to display.")
        return

    st.subheader("📈 AI Recommended Charts")

    for i in range(0, len(charts), 2):
        col1, col2 = st.columns(2)

        with col1:
            create_chart(df, charts[i])

        if i + 1 < len(charts):
            with col2:
                create_chart(df, charts[i + 1])


def get_chart_takeaways(df, chart):
    """
    Generates accurate takeaway points by computing from actual data
    rather than relying on AI to guess
    """
    chart_type = chart.get("type")
    x = chart.get("x")
    y = chart.get("y")
    takeaways = []

    try:
        match chart_type:

            case "bar_chart":
                if x and y and x in df.columns and y in df.columns:
                    grouped = df.groupby(x)[y].sum().sort_values(ascending=False)
                    top = grouped.index[0]
                    top_val = grouped.iloc[0]
                    bottom = grouped.index[-1]
                    bottom_val = grouped.iloc[-1]
                    takeaways.append(
                        f"**{top}** has the highest {y.replace('_', ' ')} at **{top_val:,.2f}**"
                    )
                    takeaways.append(
                        f"**{bottom}** has the lowest at **{bottom_val:,.2f}**"
                    )

            case "line_chart":
                if x and y and x in df.columns and y in df.columns:
                    df_sorted = df.sort_values(by=x)
                    first_val = df_sorted[y].iloc[0]
                    last_val = df_sorted[y].iloc[-1]
                    change = last_val - first_val
                    direction = "increased" if change > 0 else "decreased"
                    takeaways.append(
                        f"{y.replace('_', ' ').title()} has **{direction}** over the period "
                        f"from **{first_val:,.2f}** to **{last_val:,.2f}**"
                    )
                    peak_row = df_sorted.loc[df_sorted[y].idxmax()]
                    takeaways.append(
                        f"Peak {y.replace('_', ' ')} was **{df_sorted[y].max():,.2f}**"
                    )

            case "histogram":
                if x and x in df.columns:
                    mean_val = df[x].mean()
                    median_val = df[x].median()
                    takeaways.append(
                        f"Average {x.replace('_', ' ')} is **{mean_val:,.2f}** "
                        f"with a median of **{median_val:,.2f}**"
                    )
                    takeaways.append(
                        f"Values range from **{df[x].min():,.2f}** to **{df[x].max():,.2f}**"
                    )

            case "scatter_plot":
                if x and y and x in df.columns and y in df.columns:
                    corr = df[x].corr(df[y])
                    direction = "positive" if corr > 0 else "negative"
                    strength = "strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.3 else "weak"
                    takeaways.append(
                        f"There is a **{strength} {direction} relationship** "
                        f"between {x.replace('_', ' ')} and {y.replace('_', ' ')} "
                        f"(correlation: {corr:.2f})"
                    )

            case "pie_chart":
                if x and y and x in df.columns and y in df.columns:
                    grouped = df.groupby(x)[y].sum().sort_values(ascending=False)
                    total = grouped.sum()
                    top = grouped.index[0]
                    top_pct = (grouped.iloc[0] / total) * 100
                    takeaways.append(
                        f"**{top}** makes up the largest share at **{top_pct:.1f}%** of total"
                    )
                    top2_pct = (grouped.iloc[:2].sum() / total) * 100
                    takeaways.append(
                        f"The top 2 categories account for **{top2_pct:.1f}%** of total {y.replace('_', ' ')}"
                    )

            case "heatmap":
                numeric_df = df.select_dtypes(include="number")
                if numeric_df.shape[1] >= 2:
                    corr = numeric_df.corr()
                    # Find strongest correlation 
                    corr_unstacked = corr.where(
                        ~(corr == 1.0)
                    ).abs().unstack().dropna().sort_values(ascending=False)
                    if not corr_unstacked.empty:
                        top_pair = corr_unstacked.index[0]
                        top_val = corr_unstacked.iloc[0]
                        takeaways.append(
                            f"**{top_pair[0].replace('_', ' ')}** and "
                            f"**{top_pair[1].replace('_', ' ')}** have the "
                            f"strongest relationship (correlation: {top_val:.2f})"
                        )

    except Exception as e:
        print(f"Could not compute takeaways: {e}")

    return takeaways