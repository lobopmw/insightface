from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "guia_operacional_monitoramento_professor.pdf"


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
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionCustom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=colors.HexColor("#183153"),
            spaceBefore=10,
            spaceAfter=8,
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
    story.append(Paragraph("Guia Operacional do Monitoramento e Armazenamento no Banco", styles["TitleCustom"]))
    story.append(
        Paragraph(
            f"Documento gerado em {date.today().strftime('%d/%m/%Y')} com base no fluxo atual implementado no sistema.",
            styles["SmallCustom"],
        )
    )
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("1. Visao geral do armazenamento", styles["SectionCustom"]))
    add_bullets(
        story,
        styles,
        [
            "Os usuarios da aplicacao ficam na tabela users.",
            "Os professores ficam vinculados em teachers e sao associados a disciplinas e turmas em teacher_subject_class.",
            "Cada aluno fica registrado na tabela students com id/hash, nome, matricula, turma e status ativo.",
            "As fotos do aluno ficam em data/alunos/<hash>/<pose>/ no filesystem local.",
            "O mapeamento logico nome + matricula + hash fica em data/mapeamento_alunos.csv.",
            "Os embeddings faciais consolidados ficam nos arquivos locais data/embeddings.npy e data/names.pkl e tambem na tabela face_embeddings.",
            "Cada sessao de aula monitorada fica na tabela monitoring_sessions.",
            "Os comportamentos persistidos ficam na tabela behavior_episode, um registro por episodio validado e nao por frame.",
        ],
    )

    story.append(Paragraph("2. Como o aluno e cadastrado hoje", styles["SectionCustom"]))
    add_bullets(
        story,
        styles,
        [
            "O professor entra em Cadastro de Alunos e escolhe a turma que sera vinculada ao aluno.",
            "Ao preencher nome e matricula, o sistema gera um hash unico do aluno.",
            "Esse hash e salvo/atualizado em data/mapeamento_alunos.csv.",
            "Nesse mesmo momento o sistema faz upsert do aluno em students com id=hash, nome, matricula e class_id.",
            "O sistema cria as pastas de poses em data/alunos/<hash>/frontal, lateral_direita, lateral_esquerda e cabeca_baixa.",
            "Durante a captura, as fotos sao gravadas nessas pastas locais e passam a compor a base visual do aluno.",
        ],
    )

    story.append(Paragraph("3. Como as imagens viram reconhecimento facial", styles["SectionCustom"]))
    add_bullets(
        story,
        styles,
        [
            "Depois de capturar as imagens, e necessario executar o processo de embeddings: python src/register_face_multi_images_avg.py.",
            "O script le data/mapeamento_alunos.csv para descobrir quais alunos existem e qual pasta de imagens pertence a cada um.",
            "Para cada imagem valida, o InsightFace extrai o embedding facial.",
            "O sistema calcula um embedding medio por aluno.",
            "Esse embedding medio e salvo localmente em data/embeddings.npy e data/names.pkl.",
            "O mesmo embedding medio e enviado ao PostgreSQL na tabela face_embeddings com student_hash, nome, matricula, embedding e updated_at.",
            "No monitoramento em tempo real, o sistema tenta carregar os embeddings primeiro do banco. Se falhar, usa os arquivos locais como fallback.",
        ],
    )

    story.append(Paragraph("4. O que o professor precisa antes de iniciar o monitoramento", styles["SectionCustom"]))
    add_bullets(
        story,
        styles,
        [
            "Ter um usuario valido com perfil professor e vinculo correto com a disciplina e a turma.",
            "Ter os alunos cadastrados na tela Cadastro de Alunos.",
            "Ter concluido a captura das fotos por pose dos alunos.",
            "Ter executado o processo de geracao de embeddings apos qualquer novo cadastro ou alteracao de fotos.",
            "Confirmar que a camera/RTSP e o relay estao funcionando.",
            "Selecionar na tela Monitoramento a disciplina e a turma corretas do professor autenticado.",
        ],
    )

    story.append(Paragraph("5. O que acontece quando o professor inicia o monitoramento", styles["SectionCustom"]))
    add_bullets(
        story,
        styles,
        [
            "Ao clicar em Iniciar monitoramento, o sistema cria uma linha em monitoring_sessions com teacher_id, subject_id, class_id, session_date, start_time e status=em_andamento.",
            "A aplicacao carrega os alunos visiveis naquele escopo e monta um lookup para associar nomes reconhecidos aos ids dos alunos.",
            "O runtime inicia o stream de video, a deteccao pose/facial e o gerenciador de episodios por aluno.",
            "A classificacao de comportamento roda continuamente, mas a persistencia nao acontece por frame.",
        ],
    )

    story.append(Paragraph("6. Como os comportamentos sao gravados por aluno", styles["SectionCustom"]))
    add_bullets(
        story,
        styles,
        [
            "Cada aluno reconhecido possui um estado em memoria controlado por BehaviorEpisodeManager.",
            "Quando o comportamento muda, a mudanca precisa permanecer estavel por tempo ou por quantidade minima de frames para virar um episodio valido.",
            "No fluxo atual, a estabilidade padrao esta configurada em 2.0 segundos ou 15 frames.",
            "Quando a mudanca e validada, o episodio anterior e fechado e enviado para insert_behavior_episode.",
            "A tabela behavior_episode recebe: monitoring_session_id, student_id, school, discipline, teacher, id_student, student, behavior, start_time, end_time, duration_seconds, date e source.",
            "Isso significa que o banco armazena intervalos de comportamento por aluno, com inicio e fim, e nao contagem frame a frame.",
        ],
    )

    story.append(Paragraph("7. O que acontece ao encerrar a sessao", styles["SectionCustom"]))
    add_bullets(
        story,
        styles,
        [
            "Ao clicar em Encerrar monitoramento, o sistema faz flush dos episodios ainda abertos em memoria.",
            "Esses episodios finais tambem sao gravados em behavior_episode com o horario de encerramento da sessao.",
            "Depois disso, a sessao correspondente em monitoring_sessions recebe end_time e status=encerrada.",
            "Os dados passam a ficar disponiveis para graficos e relatorios filtrados por professor, turma, disciplina, aluno e periodo.",
        ],
    )

    story.append(Paragraph("8. Procedimento operacional completo para o professor", styles["SectionCustom"]))
    numbered_rows = [
        ["1", "Entrar no sistema com usuario professor corretamente vinculado a disciplina e turma."],
        ["2", "Abrir Cadastro de Alunos e selecionar a turma do aluno."],
        ["3", "Informar nome e matricula do aluno."],
        ["4", "Capturar as imagens exigidas nas poses orientadas pelo sistema."],
        ["5", "Finalizar o cadastro para gravar students, mapeamento e imagens locais."],
        ["6", "Executar o processo de embeddings para atualizar face_embeddings e os arquivos locais de reconhecimento."],
        ["7", "Abrir Monitoramento e selecionar disciplina e turma."],
        ["8", "Clicar em Iniciar monitoramento para abrir a sessao em monitoring_sessions."],
        ["9", "Acompanhar a aula. O sistema reconhece o aluno, classifica o comportamento e grava episodios validos em behavior_episode."],
        ["10", "Ao final da aula, clicar em Encerrar monitoramento para fechar os episodios pendentes e encerrar a sessao no banco."],
        ["11", "Consultar graficos e relatorios para analise posterior."],
    ]
    table = Table(numbered_rows, colWidths=[1.1 * cm, 14.7 * cm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    story.append(table)

    story.append(Paragraph("9. Observacoes importantes", styles["SectionCustom"]))
    add_bullets(
        story,
        styles,
        [
            "Se o professor cadastrar um novo aluno e nao atualizar os embeddings, o reconhecimento facial desse aluno nao ficara confiavel no monitoramento.",
            "O monitoramento depende do vinculo professor-disciplina-turma ja existir no banco.",
            "O banco hoje registra o comportamento por episodio consolidado, o que reduz ruido e evita gravacao excessiva.",
            "Em caso de queda de energia, travamento ou parada abrupta, o ideal e encerrar a sessao corretamente para garantir o flush final dos episodios em memoria.",
        ],
    )

    doc.build(story)


if __name__ == "__main__":
    build_pdf(OUTPUT_PATH)
    print(OUTPUT_PATH)
