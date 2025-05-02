#!/usr/bin/env python3
"""
Gera um relatório de horas (planejadas x reais) por usuário.
Uso:
    streamlit run main.py

O aplicativo permite fazer upload do arquivo CSV e visualiza o relatório de horas.
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _parse_hours(raw) -> float:
    """
    Converte valores como "08:30", "4", "4:00", "4,75" → horas (float).
    Valores vazios viram 0.
    """
    if pd.isna(raw):
        return 0.0
    raw = str(raw).strip()
    if raw == "":
        return 0.0

    # Formato HH:MM
    if re.match(r"^\d+:\d+$", raw):
        h, m = map(int, raw.split(":"))
        return h + m / 60

    # Troca vírgula por ponto (ex.: 4,75 → 4.75)
    raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        # Extrai o primeiro número que encontrar
        num = re.search(r"(\d+[\.,]?\d*)", raw)
        return float(num.group(1).replace(",", ".")) if num else 0.0


def summarize_hours(csv_file) -> pd.DataFrame:
    """Lê o CSV e devolve um DataFrame com o total de horas por usuário."""
    df = pd.read_csv(csv_file, encoding="latin1")

    # Converte texto → números
    df["planned_hours"] = df["u_horas_planejadas"].apply(_parse_hours)
    df["real_hours"] = df["u_horas_reais"].apply(_parse_hours)

    # Considera apenas tarefas finalizadas
    done = df[df["state"].str.lower() == "concluído"]

    # Soma por pessoa
    resumo = (
        done.groupby("assigned_to", as_index=False)
        .agg(
            total_planned_hours=("planned_hours", "sum"),
            total_real_hours=("real_hours", "sum"),
        )
        .sort_values("assigned_to")
    )

    # Calcula a diferença entre horas reais e planejadas
    resumo["difference"] = resumo["total_real_hours"] - resumo["total_planned_hours"]

    # Calcula a precisão da estimativa (quanto mais próximo de 100%, melhor)
    resumo["estimation_accuracy"] = (
        (
            100
            - abs(
                resumo["difference"]
                / resumo["total_planned_hours"].replace(0, float("nan"))
                * 100
            )
        )
        .fillna(0)
        .clip(0, 100)
    )

    return resumo


def get_task_status_summary(df):
    """Gera um resumo de tarefas por status."""
    status_counts = df["state"].value_counts().reset_index()
    status_counts.columns = ["Status", "Quantidade"]
    return status_counts


def get_epic_summary(df):
    """Gera um resumo de horas por epic."""
    df["planned_hours"] = df["u_horas_planejadas"].apply(_parse_hours)
    df["real_hours"] = df["u_horas_reais"].apply(_parse_hours)

    # Remove empty epics
    df_with_epic = df[df["story.epic"].notna() & (df["story.epic"] != "")]

    epic_summary = (
        df_with_epic.groupby("story.epic", as_index=False)
        .agg(
            num_tasks=("number", "count"),
            total_planned_hours=("planned_hours", "sum"),
            total_real_hours=("real_hours", "sum"),
            pct_completed=(
                "state",
                lambda x: (x.str.lower() == "concluído").mean() * 100,
            ),
        )
        .sort_values("num_tasks", ascending=False)
    )

    epic_summary["difference"] = (
        epic_summary["total_real_hours"] - epic_summary["total_planned_hours"]
    )
    return epic_summary


def get_sprint_summary(df):
    """Gera um resumo de horas por sprint."""
    df["planned_hours"] = df["u_horas_planejadas"].apply(_parse_hours)
    df["real_hours"] = df["u_horas_reais"].apply(_parse_hours)

    sprint_summary = (
        df.groupby("story.sprint", as_index=False)
        .agg(
            num_tasks=("number", "count"),
            total_planned_hours=("planned_hours", "sum"),
            total_real_hours=("real_hours", "sum"),
            pct_completed=(
                "state",
                lambda x: (x.str.lower() == "concluído").mean() * 100,
            ),
        )
        .sort_values("story.sprint")
    )

    sprint_summary["difference"] = (
        sprint_summary["total_real_hours"] - sprint_summary["total_planned_hours"]
    )
    return sprint_summary


def get_daily_workload(df):
    """Analisa a carga de trabalho ao longo do período da sprint."""
    # Verificar se as colunas de data existem e têm dados
    if (
        "story.sprint.start_date" not in df.columns
        or "story.sprint.end_date" not in df.columns
    ):
        return None

    # Converter colunas de data para datetime
    try:
        df["start_date"] = pd.to_datetime(
            df["story.sprint.start_date"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )
        df["end_date"] = pd.to_datetime(
            df["story.sprint.end_date"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )

        # Se não conseguiu converter nenhuma data, retorna None
        if df["start_date"].isna().all() or df["end_date"].isna().all():
            return None

        # Filtrar registros com datas válidas
        df_with_dates = df.dropna(subset=["start_date", "end_date"])
        if len(df_with_dates) == 0:
            return None

        # Criar um DataFrame com dias entre início e fim da sprint
        start_date = df_with_dates["start_date"].min()
        end_date = df_with_dates["end_date"].max()

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Distribuir horas pelas datas (simplificado - distribuição uniforme)
        daily_load = pd.DataFrame(index=date_range)
        daily_load["planned_hours"] = 0.0
        daily_load["real_hours"] = 0.0

        # Para cada tarefa, distribua as horas pelos dias da sprint
        for _, row in df_with_dates.iterrows():
            task_days = (row["end_date"] - row["start_date"]).days + 1
            if task_days > 0:
                daily_planned = _parse_hours(row["u_horas_planejadas"]) / task_days
                daily_real = _parse_hours(row["u_horas_reais"]) / task_days

                task_dates = pd.date_range(
                    start=row["start_date"], end=row["end_date"], freq="D"
                )
                for date in task_dates:
                    if date in daily_load.index:
                        daily_load.at[date, "planned_hours"] = (
                            float(daily_load.at[date, "planned_hours"]) + daily_planned
                        )
                        daily_load.at[date, "real_hours"] = (
                            float(daily_load.at[date, "real_hours"]) + daily_real
                        )

        daily_load = daily_load.reset_index()
        daily_load.rename(columns={"index": "date"}, inplace=True)
        return daily_load

    except Exception:
        return None


def prepare_tasks_data(df):
    """Prepara os dados para o explorador de tarefas."""
    # Adiciona colunas necessárias
    df = df.copy()
    df["planned_hours"] = df["u_horas_planejadas"].apply(_parse_hours)
    df["real_hours"] = df["u_horas_reais"].apply(_parse_hours)

    # Calcula a diferença entre horas reais e planejadas
    df["difference"] = df["real_hours"] - df["planned_hours"]

    # Calcula eficiência (real / planejado)
    # Evita divisão por zero
    df["efficiency"] = (
        df["planned_hours"] / df["real_hours"].replace(0, float("nan"))
    ).fillna(0)
    df["efficiency"] = df["efficiency"].clip(0, 2)  # limita entre 0 e 200%

    # Flag para tarefas sem estimativa
    df["has_estimate"] = df["planned_hours"] > 0

    # Interpreta datas onde disponíveis
    if (
        "story.sprint.start_date" in df.columns
        and "story.sprint.end_date" in df.columns
    ):
        try:
            df["sprint_start_date"] = pd.to_datetime(
                df["story.sprint.start_date"],
                format="%d/%m/%Y %H:%M:%S",
                errors="coerce",
            )
            df["sprint_end_date"] = pd.to_datetime(
                df["story.sprint.end_date"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
            )
            df["sprint_duration_days"] = (
                df["sprint_end_date"] - df["sprint_start_date"]
            ).dt.days
        except Exception as e:
            st.error(f"Erro ao processar os dados da sprint: {e}")
            pass

    return df


def main():
    """Interface Streamlit para o relatório de horas."""
    st.set_page_config(page_title="Relatório de Horas", layout="wide")

    st.title("Relatório de Horas por Usuário")
    st.markdown(
        """
        Faça o upload do arquivo CSV exportado do Redmine para visualizar o relatório 
        de horas planejadas x reais por usuário.
    """
    )

    # Upload do arquivo
    uploaded_file = st.file_uploader("Escolha o arquivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Lê o arquivo CSV
            df_original = pd.read_csv(uploaded_file, encoding="latin1")
            # Reset do cursor para reutilizar o arquivo
            uploaded_file.seek(0)

            # Processar o arquivo para o relatório por usuário
            relatorio = summarize_hours(uploaded_file)

            # Preparar dados para o explorador de tarefas
            df_tasks = prepare_tasks_data(df_original)

            # Criar abas para diferentes visualizações
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                [
                    "Horas por Usuário",
                    "Status das Tarefas",
                    "Análise por Epic",
                    "Análise por Sprint",
                    "Métricas Avançadas",
                    "Explorador de Tarefas",
                ]
            )

            # Tab 1: Relatório de Horas por Usuário
            with tab1:
                st.subheader("Relatório de Horas (Tarefas Concluídas)")

                # Formatação para exibição
                df_display = relatorio.copy()
                df_display["total_planned_hours"] = df_display[
                    "total_planned_hours"
                ].map("{:.2f} h".format)
                df_display["total_real_hours"] = df_display["total_real_hours"].map(
                    "{:.2f} h".format
                )
                df_display["difference"] = df_display["difference"].map(
                    "{:.2f} h".format
                )
                df_display["estimation_accuracy"] = df_display[
                    "estimation_accuracy"
                ].map("{:.1f}%".format)

                df_display = df_display.rename(
                    columns={
                        "assigned_to": "Usuário",
                        "total_planned_hours": "Horas Planejadas",
                        "total_real_hours": "Horas Reais",
                        "difference": "Diferença (Real - Planejada)",
                        "estimation_accuracy": "Precisão da Estimativa",
                    }
                )

                st.dataframe(df_display, use_container_width=True)

                # Gráfico de barras comparativo
                st.subheader("Gráfico Comparativo")

                fig = px.bar(
                    relatorio,
                    x="assigned_to",
                    y=["total_planned_hours", "total_real_hours"],
                    barmode="group",
                    labels={
                        "assigned_to": "Usuário",
                        "total_planned_hours": "Horas Planejadas",
                        "total_real_hours": "Horas Reais",
                        "value": "Horas",
                    },
                    title="Comparação entre Horas Planejadas e Reais por Usuário",
                    color_discrete_sequence=["#1f77b4", "#ff7f0e"],
                )
                st.plotly_chart(fig, use_container_width=True)

                # Gráfico de precisão da estimativa
                st.subheader("Precisão da Estimativa por Usuário")
                fig = px.bar(
                    relatorio,
                    x="assigned_to",
                    y="estimation_accuracy",
                    labels={
                        "assigned_to": "Usuário",
                        "estimation_accuracy": "Precisão da Estimativa (%)",
                    },
                    title="Precisão da Estimativa por Usuário",
                    color="estimation_accuracy",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 100],
                )
                st.plotly_chart(fig, use_container_width=True)

            # Tab 2: Status das Tarefas
            with tab2:
                st.subheader("Distribuição de Status das Tarefas")

                # Gerar resumo de status
                status_summary = get_task_status_summary(df_original)

                # Gráfico de pizza para status
                fig = px.pie(
                    status_summary,
                    values="Quantidade",
                    names="Status",
                    title="Distribuição de Tarefas por Status",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tabela de status
                st.dataframe(status_summary, use_container_width=True)

                # Número de tarefas por usuário
                st.subheader("Tarefas por Usuário")
                tasks_by_user = df_original["assigned_to"].value_counts().reset_index()
                tasks_by_user.columns = ["Usuário", "Número de Tarefas"]

                fig = px.bar(
                    tasks_by_user,
                    x="Usuário",
                    y="Número de Tarefas",
                    title="Quantidade de Tarefas por Usuário",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Tab 3: Análise por Epic
            with tab3:
                st.subheader("Análise por Epic")

                # Gerar resumo por epic
                epic_summary = get_epic_summary(df_original)

                # Formatar para exibição
                epic_display = epic_summary.copy()
                epic_display["total_planned_hours"] = epic_display[
                    "total_planned_hours"
                ].map("{:.2f} h".format)
                epic_display["total_real_hours"] = epic_display["total_real_hours"].map(
                    "{:.2f} h".format
                )
                epic_display["difference"] = epic_display["difference"].map(
                    "{:.2f} h".format
                )
                epic_display["pct_completed"] = epic_display["pct_completed"].map(
                    "{:.1f}%".format
                )

                epic_display = epic_display.rename(
                    columns={
                        "story.epic": "Epic",
                        "num_tasks": "Número de Tarefas",
                        "total_planned_hours": "Horas Planejadas",
                        "total_real_hours": "Horas Reais",
                        "difference": "Diferença (Real - Planejada)",
                        "pct_completed": "% Concluído",
                    }
                )

                st.dataframe(epic_display, use_container_width=True)

                # Gráfico de progresso por epic
                if not epic_summary.empty:
                    fig = px.bar(
                        epic_summary.sort_values("pct_completed"),
                        x="pct_completed",
                        y="story.epic",
                        orientation="h",
                        labels={"story.epic": "Epic", "pct_completed": "% Concluído"},
                        title="Progresso por Epic (%)",
                        color="pct_completed",
                        color_continuous_scale="Blues",
                        range_color=[0, 100],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Gráfico de horas por epic
                    fig = px.bar(
                        epic_summary,
                        x="story.epic",
                        y=["total_planned_hours", "total_real_hours"],
                        barmode="group",
                        labels={
                            "story.epic": "Epic",
                            "total_planned_hours": "Horas Planejadas",
                            "total_real_hours": "Horas Reais",
                            "value": "Horas",
                        },
                        title="Horas Planejadas vs. Reais por Epic",
                        color_discrete_sequence=["#1f77b4", "#ff7f0e"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Tab 4: Análise por Sprint
            with tab4:
                st.subheader("Análise por Sprint")

                # Gerar resumo por sprint
                sprint_summary = get_sprint_summary(df_original)

                # Formatar para exibição
                sprint_display = sprint_summary.copy()
                sprint_display["total_planned_hours"] = sprint_display[
                    "total_planned_hours"
                ].map("{:.2f} h".format)
                sprint_display["total_real_hours"] = sprint_display[
                    "total_real_hours"
                ].map("{:.2f} h".format)
                sprint_display["difference"] = sprint_display["difference"].map(
                    "{:.2f} h".format
                )
                sprint_display["pct_completed"] = sprint_display["pct_completed"].map(
                    "{:.1f}%".format
                )

                sprint_display = sprint_display.rename(
                    columns={
                        "story.sprint": "Sprint",
                        "num_tasks": "Número de Tarefas",
                        "total_planned_hours": "Horas Planejadas",
                        "total_real_hours": "Horas Reais",
                        "difference": "Diferença (Real - Planejada)",
                        "pct_completed": "% Concluído",
                    }
                )

                st.dataframe(sprint_display, use_container_width=True)

                # Gráfico de velocidade da sprint
                if not sprint_summary.empty:
                    fig = px.line(
                        sprint_summary,
                        x="story.sprint",
                        y=["total_planned_hours", "total_real_hours"],
                        markers=True,
                        labels={
                            "story.sprint": "Sprint",
                            "value": "Horas",
                            "variable": "Tipo",
                        },
                        title="Velocidade da Sprint (Horas Planejadas vs. Reais)",
                    )
                    fig.update_layout(
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Gráfico de % completado por sprint
                    fig = px.bar(
                        sprint_summary,
                        x="story.sprint",
                        y="pct_completed",
                        labels={
                            "story.sprint": "Sprint",
                            "pct_completed": "% Concluído",
                        },
                        title="Percentual de Conclusão por Sprint",
                        color="pct_completed",
                        color_continuous_scale="Greens",
                        range_color=[0, 100],
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Tab 5: Métricas Avançadas
            with tab5:
                st.subheader("Métricas Avançadas")

                # Colunas para métricas gerais
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Média de precisão de estimativas
                    avg_accuracy = relatorio["estimation_accuracy"].mean()
                    st.metric(
                        "Média de Precisão de Estimativas",
                        f"{avg_accuracy:.1f}%",
                        delta=(
                            f"{avg_accuracy - 80:.1f}%" if avg_accuracy != 80 else None
                        ),
                        delta_color="normal",
                    )

                with col2:
                    # Proporção de tarefas concluídas
                    tasks_done = (df_original["state"].str.lower() == "concluído").sum()
                    total_tasks = len(df_original)
                    pct_done = (
                        (tasks_done / total_tasks) * 100 if total_tasks > 0 else 0
                    )
                    st.metric(
                        "Tarefas Concluídas",
                        f"{pct_done:.1f}%",
                        f"{tasks_done} de {total_tasks}",
                    )

                with col3:
                    # Diferença total entre planejado e real
                    total_planned = relatorio["total_planned_hours"].sum()
                    total_real = relatorio["total_real_hours"].sum()
                    diff = total_real - total_planned
                    st.metric(
                        "Diferença Total (Real - Planejado)",
                        f"{diff:.2f} h",
                        delta=f"{diff:.2f} h",
                        delta_color="inverse",
                    )

                # Carga diária de trabalho
                st.subheader("Carga Diária de Trabalho")
                daily_workload = get_daily_workload(df_original)

                if daily_workload is not None:
                    # Formatar datas para exibição
                    daily_workload["date_str"] = daily_workload["date"].dt.strftime(
                        "%d/%m/%Y"
                    )

                    # Gráfico de linha para carga diária
                    fig = px.line(
                        daily_workload,
                        x="date",
                        y=["planned_hours", "real_hours"],
                        markers=True,
                        labels={"date": "Data", "value": "Horas", "variable": "Tipo"},
                        title="Distribuição da Carga de Trabalho Diária",
                    )
                    fig.update_layout(xaxis_title="Data")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "Não foi possível calcular a carga diária de trabalho. Verifique se o arquivo CSV contém as datas de início e fim da sprint."
                    )

                # Top contribuidores
                st.subheader("Top Contribuidores")
                top_contributors = relatorio.sort_values(
                    "total_real_hours", ascending=False
                ).head(5)

                if not top_contributors.empty:
                    fig = px.bar(
                        top_contributors,
                        x="assigned_to",
                        y="total_real_hours",
                        labels={
                            "assigned_to": "Usuário",
                            "total_real_hours": "Horas Reais",
                        },
                        title="Top 5 Contribuidores",
                        color="total_real_hours",
                        color_continuous_scale="Viridis",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Tab 6: Explorador de Tarefas
            with tab6:
                st.subheader("Explorador de Tarefas")

                # Usar uma coluna para organizar o layout (filtros à esquerda, conteúdo à direita)
                filter_col, content_col = st.columns([1, 3])

                with filter_col:
                    st.markdown("### Filtros")

                    # Preparar opções para filtros
                    status_options = ["Todos"] + sorted(
                        df_tasks["state"].unique().tolist()
                    )
                    epic_options = ["Todos"] + sorted(
                        [
                            epic
                            for epic in df_tasks["story.epic"].unique()
                            if pd.notna(epic) and epic != ""
                        ]
                    )
                    person_options = ["Todos"] + sorted(
                        df_tasks["assigned_to"].unique().tolist()
                    )
                    sprint_options = ["Todos"] + sorted(
                        df_tasks["story.sprint"].unique().tolist()
                    )

                    # Adicionar os filtros
                    selected_status = st.selectbox("Status", status_options)
                    selected_epic = st.selectbox("Epic", epic_options)
                    selected_person = st.selectbox("Pessoa", person_options)
                    selected_sprint = st.selectbox("Sprint", sprint_options)

                    # Filtrar por range de horas planejadas
                    max_planned = float(df_tasks["planned_hours"].max())
                    planned_range = st.slider(
                        "Horas Planejadas",
                        0.0,
                        (
                            max_planned
                            if pd.notna(max_planned) and max_planned > 0
                            else 100.0
                        ),
                        (
                            0.0,
                            (
                                max_planned
                                if pd.notna(max_planned) and max_planned > 0
                                else 100.0
                            ),
                        ),
                    )

                # Aplicar filtros
                filtered_df = df_tasks.copy()

                if selected_status != "Todos":
                    filtered_df = filtered_df[filtered_df["state"] == selected_status]

                if selected_epic != "Todos":
                    filtered_df = filtered_df[
                        filtered_df["story.epic"] == selected_epic
                    ]

                if selected_person != "Todos":
                    filtered_df = filtered_df[
                        filtered_df["assigned_to"] == selected_person
                    ]

                if selected_sprint != "Todos":
                    filtered_df = filtered_df[
                        filtered_df["story.sprint"] == selected_sprint
                    ]

                filtered_df = filtered_df[
                    (filtered_df["planned_hours"] >= planned_range[0])
                    & (filtered_df["planned_hours"] <= planned_range[1])
                ]

                with content_col:
                    # Exibir métricas interessantes baseadas na filtragem
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    with metrics_col1:
                        task_count = len(filtered_df)
                        st.metric("Total de Tarefas", task_count)

                    with metrics_col2:
                        completed_tasks = filtered_df[
                            filtered_df["state"].str.lower() == "concluído"
                        ].shape[0]
                        completion_rate = (
                            (completed_tasks / task_count * 100)
                            if task_count > 0
                            else 0
                        )
                        st.metric(
                            "Taxa de Conclusão",
                            f"{completion_rate:.1f}%",
                            f"{completed_tasks} de {task_count}",
                        )

                    with metrics_col3:
                        missing_estimates = filtered_df[
                            filtered_df["planned_hours"] == 0
                        ].shape[0]
                        missing_rate = (
                            (missing_estimates / task_count * 100)
                            if task_count > 0
                            else 0
                        )
                        st.metric(
                            "Tarefas sem Estimativa",
                            f"{missing_rate:.1f}%",
                            f"{missing_estimates} de {task_count}",
                            delta_color="inverse",
                        )

                    # Gráficos específicos para a visualização filtrada
                    if len(filtered_df) > 0:
                        # Gráfico de Eficiência para tarefas concluídas
                        completed_tasks_df = filtered_df[
                            filtered_df["state"].str.lower() == "concluído"
                        ].copy()
                        completed_tasks_df = completed_tasks_df[
                            completed_tasks_df["planned_hours"] > 0
                        ]

                        if len(completed_tasks_df) > 0:
                            efficiency_col1, efficiency_col2 = st.columns(2)

                            with efficiency_col1:
                                st.subheader("Eficiência por Tarefa (Concluídas)")

                                fig = px.scatter(
                                    completed_tasks_df,
                                    x="planned_hours",
                                    y="real_hours",
                                    color="efficiency",
                                    hover_name="short_description",
                                    color_continuous_scale="RdYlGn_r",
                                    labels={
                                        "planned_hours": "Horas Planejadas",
                                        "real_hours": "Horas Reais",
                                        "efficiency": "Eficiência",
                                    },
                                    title="Relação entre Horas Planejadas e Reais",
                                )

                                # Adicionar linha de referência (planejado = real)
                                max_hours = max(
                                    completed_tasks_df["planned_hours"].max(),
                                    completed_tasks_df["real_hours"].max(),
                                )
                                fig.add_trace(
                                    go.Scatter(
                                        x=[0, max_hours],
                                        y=[0, max_hours],
                                        mode="lines",
                                        line=dict(color="gray", dash="dash"),
                                        name="Ideal (Planejado = Real)",
                                    )
                                )

                                st.plotly_chart(fig, use_container_width=True)

                            with efficiency_col2:
                                st.subheader("Distribuição da Eficiência")

                                # Categorizar eficiência
                                def categorize_efficiency(eff):
                                    if eff == 0:
                                        return "Sem estimativa"
                                    elif eff < 0.5:
                                        return "Muito abaixo (>200%)"
                                    elif eff < 0.8:
                                        return "Abaixo (125-200%)"
                                    elif eff < 1.25:
                                        return "Adequada (80-125%)"
                                    elif eff < 2:
                                        return "Acima (50-80%)"
                                    else:
                                        return "Muito acima (<50%)"

                                completed_tasks_df["efficiency_category"] = (
                                    completed_tasks_df["efficiency"].apply(
                                        categorize_efficiency
                                    )
                                )

                                category_order = [
                                    "Sem estimativa",
                                    "Muito abaixo (>200%)",
                                    "Abaixo (125-200%)",
                                    "Adequada (80-125%)",
                                    "Acima (50-80%)",
                                    "Muito acima (<50%)",
                                ]

                                # Count por categoria
                                efficiency_counts = (
                                    completed_tasks_df["efficiency_category"]
                                    .value_counts()
                                    .reset_index()
                                )
                                efficiency_counts.columns = ["Categoria", "Quantidade"]

                                # Reordenar categorias
                                efficiency_counts["Categoria"] = pd.Categorical(
                                    efficiency_counts["Categoria"],
                                    categories=category_order,
                                    ordered=True,
                                )
                                efficiency_counts = efficiency_counts.sort_values(
                                    "Categoria"
                                )

                                fig = px.bar(
                                    efficiency_counts,
                                    x="Categoria",
                                    y="Quantidade",
                                    title="Distribuição da Eficiência das Estimativas",
                                    color="Categoria",
                                    color_discrete_map={
                                        "Sem estimativa": "#808080",
                                        "Muito abaixo (>200%)": "#d62728",
                                        "Abaixo (125-200%)": "#ff7f0e",
                                        "Adequada (80-125%)": "#2ca02c",
                                        "Acima (50-80%)": "#ff7f0e",
                                        "Muito acima (<50%)": "#d62728",
                                    },
                                )

                                st.plotly_chart(fig, use_container_width=True)

                    # Tabela completa com todas as tarefas filtradas
                    st.subheader("Lista de Tarefas")

                    # Colunas a serem exibidas
                    display_columns = [
                        "number",
                        "short_description",
                        "story.sprint",
                        "story.epic",
                        "assigned_to",
                        "state",
                        "planned_hours",
                        "real_hours",
                        "difference",
                    ]

                    # Verificar se as colunas existem e criar um DataFrame para exibição
                    display_columns = [
                        col for col in display_columns if col in filtered_df.columns
                    ]
                    display_df = filtered_df[display_columns].copy()

                    # Renomear colunas para exibição
                    column_names = {
                        "number": "Número",
                        "short_description": "Descrição",
                        "story.sprint": "Sprint",
                        "story.epic": "Epic",
                        "assigned_to": "Responsável",
                        "state": "Status",
                        "planned_hours": "Horas Planejadas",
                        "real_hours": "Horas Reais",
                        "difference": "Diferença",
                    }

                    # Aplicar renomeação apenas para colunas que existem
                    rename_cols = {
                        k: v for k, v in column_names.items() if k in display_df.columns
                    }
                    display_df = display_df.rename(columns=rename_cols)

                    # Formatar colunas numéricas
                    for col in ["Horas Planejadas", "Horas Reais", "Diferença"]:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].map("{:.2f}".format)

                    # Exibir tabela
                    st.dataframe(display_df, use_container_width=True)

                    # Download das tarefas filtradas como CSV
                    if not filtered_df.empty:
                        filtered_csv = filtered_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Baixar tarefas filtradas como CSV",
                            data=filtered_csv,
                            file_name="tarefas_filtradas.csv",
                            mime="text/csv",
                        )

            # Download do relatório como CSV
            csv_export = relatorio.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Baixar relatório como CSV",
                data=csv_export,
                file_name="relatorio_horas.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            st.error(
                "Verifique se o formato do arquivo CSV é compatível com o esperado."
            )


if __name__ == "__main__":
    main()
