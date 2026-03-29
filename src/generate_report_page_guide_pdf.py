from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "guia_pagina_relatorios_professor.pdf"


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=6,
            textColor=colors.HexColor("#1F2937"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionCustom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=colors.HexColor("#183153"),
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#475569"),
            spaceAfter=4,
        )
    )
    return styles


def bullet(text: str) -> str:
    return f"&bull; {text}"


def add_bullets(story, styles, items):
    for item in items:
        story.append(Paragraph(bullet(item), styles["BodyCustom"]))


def add_section(story, styles, title: str, items: list[str]):
    story.append(Paragraph(title, styles["SectionCustom"]))
    add_bullets(story, styles, items)


def build_summary_table():
    rows = [
        ["Item da tela", "O que transmite ao professor", "Como apoia a leitura do aluno"],
        [
            "Filtros",
            "Recorte por professor, disciplina, turma, aluno e periodo.",
            "Garante que a analise esteja contextualizada antes da leitura dos indicadores.",
        ],
        [
            "Acoes do topo",
            "Gera PDF e exporta CSVs de resumo e episodios.",
            "Transforma a leitura em material compartilhavel, auditavel e arquivavel.",
        ],
        [
            "KPIs",
            "Mostram volume de episodios, dias com registros, comportamento predominante e duracao acumulada.",
            "Entregam uma leitura executiva imediata do periodo analisado.",
        ],
        [
            "Aba Visao Geral",
            "Resume o caso, interpreta o periodo e destaca pontos de atencao.",
            "Ajuda o professor a entender o quadro geral antes de entrar nos detalhes.",
        ],
        [
            "Aba Frequencia e Duracao",
            "Mostra distribuicao por comportamento, recorrencia e tempo estimado.",
            "Ajuda a comparar o peso relativo dos comportamentos observados.",
        ],
        [
            "Aba Distribuicao Temporal",
            "Mostra linha do tempo dos episodios ao longo da aula.",
            "Explica quando cada padrao apareceu e em que concentracao.",
        ],
        [
            "Aba Comparacoes",
            "Compara com periodo anterior e com recortes internos do intervalo.",
            "Ajuda a perceber aumento, reducao ou estabilidade observacional.",
        ],
        [
            "Detalhes e Exportacao",
            "Abre tabelas extensas e repete as saidas em CSV e PDF.",
            "Permite leitura aprofundada, rastreio cronologico e compartilhamento formal.",
        ],
    ]
    table = Table(rows, colWidths=[4.1 * cm, 6.0 * cm, 6.0 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_step_table():
    rows = [
        ["Passo", "Fluxo de leitura da pagina de relatorios"],
        ["1", "O professor chega na tela Relatorios Observacionais e identifica o objetivo geral da pagina."],
        ["2", "Seleciona professor, disciplina, turma, aluno e periodo para definir o recorte analitico."],
        ["3", "O sistema consolida os episodios validos do aluno dentro do intervalo e monta os indicadores-base."],
        ["4", "As acoes do topo ja oferecem PDF e CSVs para formalizacao e compartilhamento."],
        ["5", "Os KPIs entregam um resumo executivo rapido do caso antes da leitura detalhada."],
        ["6", "A aba Visao Geral contextualiza o periodo com sintese, interpretacao e pontos de atencao."],
        ["7", "A aba Frequencia e Duracao aprofunda o peso relativo de cada comportamento observado."],
        ["8", "A aba Distribuicao Temporal mostra quando os episodios aconteceram ao longo da aula."],
        ["9", "A aba Comparacoes ajuda a perceber mudancas em relacao ao periodo anterior ou entre recortes do intervalo."],
        ["10", "A aba Detalhes e Exportacao abre tabelas extensas, sintese completa, notas metodologicas e os downloads finais."],
    ]
    table = Table(rows, colWidths=[1.2 * cm, 14.6 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.2),
                ("LEADING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    return table


def build_pdf(output_path: Path):
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.6 * cm,
    )

    story = []
    story.append(Paragraph("Relatorio da Pagina de Relatorios para o Professor", styles["TitleCustom"]))
    story.append(
        Paragraph(
            (
                f"Documento gerado em {date.today().strftime('%d/%m/%Y')} com base no fluxo atual implementado em "
                "src/ui/report_page.py e src/services/report_service.py."
            ),
            styles["SmallCustom"],
        )
    )
    story.append(
        Paragraph(
            (
                "Objetivo: explicar, de forma detalhada, como a pagina Relatorios Observacionais transmite ao professor "
                "os dados do aluno, desde a visao geral executiva ate os detalhes tecnicos e as exportacoes."
            ),
            styles["BodyCustom"],
        )
    )
    story.append(Spacer(1, 0.25 * cm))

    add_section(
        story,
        styles,
        "1. Visao geral da pagina",
        [
            "A pagina foi desenhada para transformar episodios observacionais brutos em leitura pedagogica organizada por aluno.",
            "A transmissao da informacao acontece em camadas: primeiro o recorte, depois o resumo executivo, em seguida os graficos e por fim os detalhes tabulares e as exportacoes.",
            "A pagina prioriza leitura progressiva: o professor nao precisa abrir tudo de uma vez, mas pode aprofundar conforme a necessidade.",
            "As acoes de exportacao aparecem no topo e tambem na ultima aba, reforcando que a analise pode ser compartilhada ou arquivada.",
        ],
    )

    add_section(
        story,
        styles,
        "2. Como os dados chegam ate a pagina",
        [
            "A interface usa get_available_filters e get_available_students para descobrir quais opcoes de professor, disciplina, turma e aluno podem ser mostradas ao usuario autenticado.",
            "Quando o professor define o recorte, a funcao generate_report_data busca os episodios observacionais persistidos e monta a base analitica do periodo.",
            "Antes de exibir, os episodios passam por preparacao: datas e horarios sao convertidos, duracao e normalizada, labels sao padronizadas e campos ausentes recebem valores substitutos.",
            "A partir dessa base, o sistema calcula resumo por comportamento, distribuicao diaria, linha do tempo, comparacao com periodo anterior, consistencia comportamental, dias de pico e metricas principais.",
            "Isso significa que a pagina nao apenas mostra registros crus: ela reorganiza os dados em formatos que ajudam a leitura docente.",
        ],
    )

    story.append(Paragraph("3. Mapa geral da transmissao de informacoes", styles["SectionCustom"]))
    story.append(build_summary_table())

    add_section(
        story,
        styles,
        "4. Filtros do relatorio: como o professor define o contexto",
        [
            "O primeiro bloco da pagina e o painel de filtros. Ele transmite ao professor que toda leitura depende de contexto e de recorte temporal.",
            "Professor: no perfil admin, o sistema permite escolher um docente especifico; no perfil professor, o campo aparece preenchido e bloqueado, reforcando o escopo do proprio usuario.",
            "Disciplina: afunila a analise para a materia selecionada, evitando leitura misturada de contextos pedagogicos diferentes.",
            "Turma: restringe a visualizacao a uma turma especifica quando necessario.",
            "Aluno: define o sujeito da analise. A pagina so prossegue se houver alunos com episodios no contexto selecionado.",
            "Periodo: pode ser diario, semanal ou mensal. Em modo diario o professor escolhe uma data; nos demais modos escolhe um intervalo.",
            "Se nao houver registros para o recorte, a interface informa isso explicitamente. No modo diario, a tela ainda tenta informar a ultima data disponivel para aquele aluno.",
        ],
    )

    add_section(
        story,
        styles,
        "5. Acoes do topo: o que a pagina disponibiliza antes mesmo das abas",
        [
            "Assim que a base do relatorio e montada, a pagina exibe um bloco de saida do relatorio com tres acoes principais.",
            "Gerar PDF: produz a versao formal do relatorio observacional, adequada para arquivo, compartilhamento ou evidencia documental.",
            "Exportar resumo: entrega um CSV com a tabela consolidada de frequencias por comportamento.",
            "Exportar episodios: entrega um CSV cronologico com os episodios observados, incluindo inicio, fim, comportamento, duracao, disciplina, professor e origem.",
            "Esse desenho transmite ao professor que a tela nao serve apenas para leitura momentanea: ela tambem suporta prestacao de contas, acompanhamento e analise posterior.",
        ],
    )

    add_section(
        story,
        styles,
        "6. KPIs: leitura executiva imediata do aluno",
        [
            "A faixa de KPIs aparece antes das abas e resume o caso em quatro perguntas essenciais.",
            "Episodios: quantos episodios observacionais foram registrados no recorte selecionado.",
            "Dias com registros: em quantos dias houve observacao valida para aquele aluno no periodo.",
            "Comportamento predominante: qual padrao teve maior peso no agregado, com destaque visual proprio.",
            "Duracao acumulada: quanto tempo total foi associado aos episodios registrados.",
            "Esses quatro itens ajudam o professor a responder rapidamente: houve muita ou pouca ocorrencia, em quantos dias isso apareceu, qual comportamento predominou e com que intensidade temporal.",
        ],
    )

    add_section(
        story,
        styles,
        "7. Aba Visao Geral: da leitura executiva para a interpretacao",
        [
            "A aba Visao Geral traduz os indicadores em linguagem mais inteligivel para o professor.",
            "Resumo Executivo: sintetiza o periodo analisado, quantidade de episodios, dias com registros, duracao acumulada e comportamento com maior frequencia.",
            "Leitura Interpretativa: apresenta uma interpretacao observacional que tenta contextualizar distribuicao, recorrencia e padroes predominantes, sempre com ressalva metodologica.",
            "Destaques Rapidos: organiza quatro mini-cartoes com predominancia, segunda recorrencia, padrao geral e variacao principal em relacao ao periodo anterior.",
            "Pontos de Atencao: lista alertas resumidos produzidos a partir de picos, distribuicao temporal e consistencia.",
            "Essa aba e importante porque aproxima a analise estatistica de uma leitura util para acompanhamento pedagogico.",
        ],
    )

    add_section(
        story,
        styles,
        "8. Aba Frequencia e Duracao: peso relativo dos comportamentos",
        [
            "A aba Frequencia e Duracao mostra dois graficos complementares.",
            "Grafico de ocorrencia relativa: revela quantas vezes cada comportamento apareceu em relacao ao total de registros.",
            "Grafico de duracao estimada: revela quanto tempo acumulado cada comportamento ocupou.",
            "Esses dois eixos evitam leitura simplista. Um comportamento pode aparecer poucas vezes, mas ocupar muito tempo; ou aparecer muitas vezes, com duracao curta.",
            "A expander da tabela consolidada de frequencias abre o detalhamento numerico em formato tabular, facilitando conferencia e exportacao.",
        ],
    )

    add_section(
        story,
        styles,
        "9. Aba Distribuicao Temporal: quando os comportamentos aconteceram",
        [
            "A linha do tempo e o principal recurso para explicar a sequencia cronologica da aula.",
            "Cada barra representa um episodio, com inicio, fim, data e comportamento associado.",
            "A cor de cada barra ajuda a distinguir rapidamente os comportamentos ao longo do tempo.",
            "O hover do grafico adiciona detalhes como horario inicial, horario final e duracao.",
            "Abaixo do grafico, a pagina mostra uma sintese temporal curta que resume concentracoes, dispersao ao longo da aula e necessidade de leitura contextualizada.",
            "Para o professor, essa aba responde a pergunta: em que momentos da aula determinado padrao apareceu e como ele se distribuiu no tempo.",
        ],
    )

    add_section(
        story,
        styles,
        "10. Aba Comparacoes: o que mudou em relacao ao passado",
        [
            "A aba Comparacoes transmite ao professor se o padrao atual e isolado ou se se repete em relacao ao periodo imediatamente anterior.",
            "A Sintese Comparativa apresenta frases curtas com os principais movimentos identificados pelo sistema.",
            "Quando o intervalo contem mais de um recorte interno, a pagina mostra tambem um grafico de comparacao entre esses recortes.",
            "A expander de comparacao detalhada abre uma tabela com registros atuais, registros anteriores e variacoes de frequencia e duracao.",
            "Isso ajuda o professor a perceber aumento, reducao ou estabilidade dos comportamentos observados.",
        ],
    )

    add_section(
        story,
        styles,
        "11. Aba Detalhes e Exportacao: aprofundamento e rastreabilidade",
        [
            "Esta aba funciona como area de apoio analitico e de formalizacao final do relatorio.",
            "No lado esquerdo, a pagina oferece expanders com tabelas detalhadas: frequencias, resumo por faixa horaria, comparacao detalhada, resumo cronologico, consistencia comportamental e dias com maior recorrencia.",
            "No lado direito, a pagina repete as exportacoes em CSV do resumo, CSV dos episodios e PDF do relatorio.",
            "A expander de sintese analitica completa entrega o texto integral do resumo geral e da leitura interpretativa.",
            "A expander de limitacoes e nota metodologica explica as cautelas de leitura e como a duracao estimada foi calculada.",
            "Esse desenho mostra ao professor que a interface foi pensada tanto para consulta rapida quanto para analise aprofundada e registro formal.",
        ],
    )

    add_section(
        story,
        styles,
        "12. O que cada tabela complementar acrescenta",
        [
            "Tabela consolidada de frequencias: confirma numericamente o peso de cada comportamento.",
            "Resumo por faixa horaria: ajuda a identificar se ha maior concentracao de episodios em determinadas janelas do dia.",
            "Comparacao detalhada com o periodo anterior: evidencia mudancas quantitativas entre janelas equivalentes.",
            "Resumo cronologico dos episodios: entrega rastreabilidade completa, episodio por episodio.",
            "Consistencia comportamental detalhada: mostra em quantos dias cada comportamento apareceu, sua cobertura dos dias e sua classificacao de regularidade.",
            "Dias com maior recorrencia por comportamento: aponta os dias de pico que merecem leitura contextual especifica.",
        ],
    )

    add_section(
        story,
        styles,
        "13. Como a pagina apoia a tomada de decisao do professor",
        [
            "A pagina nao entrega apenas contagem de eventos. Ela organiza os dados para apoiar interpretacao, acompanhamento e comunicacao.",
            "O professor consegue sair da pergunta 'o que aconteceu?' para perguntas mais uteis: 'o que predominou?', 'quando isso aconteceu?', 'isso se repetiu?', 'foi concentrado ou regular?', 'consigo exportar para discutir com a equipe?'.",
            "A combinacao de indicadores, textos sintese, graficos e tabelas reduz o risco de leitura fragmentada.",
            "As ressalvas metodologicas impedem que a tela seja lida como diagnostico fechado; ela se apresenta como apoio observacional.",
        ],
    )

    story.append(Paragraph("14. Fluxo resumido da experiencia do professor", styles["SectionCustom"]))
    story.append(build_step_table())

    add_section(
        story,
        styles,
        "15. Conclusao",
        [
            "A pagina Relatorios Observacionais transmite a informacao do aluno em camadas bem definidas: contexto, sintese executiva, interpretacao, distribuicoes, comparacoes, detalhes e exportacao.",
            "Esse desenho favorece tanto a consulta rapida durante a rotina docente quanto a documentacao mais formal para acompanhamento pedagogico.",
            "Em termos de produto, a aba final de detalhes e exportacao fecha o ciclo da analise: tudo que foi resumido visualmente pode ser auditado, aprofundado e baixado.",
        ],
    )

    doc.build(story)


if __name__ == "__main__":
    build_pdf(OUTPUT_PATH)
    print(OUTPUT_PATH)
