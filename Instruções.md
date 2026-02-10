Dashboard de Performance Logística
Objetivo

Este projeto visa desenvolver um Dashboard de Performance Logística que permita aos gestores monitorar a eficiência das entregas, identificar gargalos e avaliar os custos logísticos por região. O objetivo é fornecer uma visão detalhada da logística da empresa e auxiliar na tomada de decisões estratégicas sobre transportadoras, prazos e otimização de rotas.

Tecnologias Utilizadas

Python: Linguagem de programação principal.

Streamlit: Framework para criar o dashboard interativo.

Pandas: Manipulação e análise dos dados.

Folium/Plotly/Altair: Para visualizações geográficas e gráficos.

Matplotlib: Para gráficos adicionais e métricas visuais.

Requisitos do Dashboard
1. KPI de Entregas no Prazo (% de entregas realizadas dentro do prazo)

Descrição:

Exibir a porcentagem de entregas realizadas dentro do prazo estipulado.

Indicador visual para facilitar a compreensão da performance.

Decisões:

Fonte dos Dados: Calcular a porcentagem de entregas dentro do prazo com base na coluna prazo_real_dias e prazo_estimado_dias.

Exibição Visual: Utilizar um medidor ou gráfico de barras para destacar a porcentagem. O valor pode ser destacado em verde para entregas no prazo e vermelho para as que passaram do prazo.

Melhores Práticas:

Interatividade: Permitir filtros para ajustar o intervalo de datas.

Responsividade: Usar uma visualização limpa, destacando o KPI.

2. Tempo Médio de Entrega por Transportadora

Descrição:

Calcular o tempo médio de entrega de cada transportadora.

Identificar transportadoras mais eficientes e aquelas com atrasos frequentes.

Decisões:

Fonte dos Dados: Calcular a média dos valores de prazo_real_dias para cada transportadora.

Exibição Visual: Utilizar um gráfico de barras para mostrar o tempo médio de entrega por transportadora. O eixo x pode ter as transportadoras, enquanto o eixo y mostra o tempo médio de entrega.

Melhores Práticas:

Filtros: Permitir a escolha do intervalo de datas e a seleção de transportadoras específicas.

Ordenação: Ordenar as transportadoras pelo tempo médio de entrega (do menor para o maior).

3. Mapa Interativo com Fluxos de Origem-Destino

Descrição:

Visualizar rotas de entrega em um mapa interativo, com origens, destinos e volume de entregas.

Indicar áreas com maior tráfego ou problemas de atraso.

Decisões:

Fonte dos Dados: Utilizar as colunas cidade_origem e cidade_destino para desenhar as rotas.

Exibição Visual: Usar uma biblioteca como Folium ou Plotly para criar o mapa interativo. Cada ponto de origem e destino será visualizado, com linhas ligando-os. O volume de entregas entre as regiões pode ser indicado com a espessura das linhas ou cores.

Melhores Práticas:

Zoom e Filtros: Permitir que o usuário altere o nível de zoom e escolha as regiões específicas que deseja analisar.

Indicadores de Problemas: Mostrar áreas com maior atraso, com cores diferentes para destacar as rotas problemáticas.

4. Custos Logísticos por Região

Descrição:

Exibir os custos logísticos de transporte por região, permitindo a análise de onde os custos são maiores e onde há oportunidades de otimização.

Decisões:

Fonte dos Dados: Utilizar a coluna custo_transporte e agregá-la por cidade_origem, cidade_destino, ou qualquer outra região geográfica desejada.

Exibição Visual: Utilizar gráficos de barras ou mapas (similar ao item anterior) para mostrar onde os custos são mais altos. O gráfico de barras pode agrupar os custos por estado ou região.

Melhores Práticas:

Agrupamento Regional: Oferecer a opção de agrupar os dados por estado ou cidade, com base nas informações disponíveis.

Análise Dinâmica: Deixar o usuário filtrar por período e transportadora.

Decisões Técnicas
Estrutura do Código

DataFrame: Utilizar o pandas para carregar e processar os dados, como mostrado na planilha fornecida. Certifique-se de tratar valores ausentes ou inconsistentes.

Funções: Dividir o código em funções reutilizáveis para as diferentes seções do Dashboard. Isso melhora a legibilidade e manutenção do código.

Streamlit: Utilize o st.write, st.metric, st.bar_chart, st.map e outros componentes do Streamlit para exibir as métricas e gráficos.

Exemplo de Fluxo do Dashboard

Visão Geral: Exibir a porcentagem de entregas no prazo e tempo médio de entrega de transportadoras, com gráficos interativos.

Detalhamento por Região: Exibir um mapa com fluxos de origem-destino, permitindo explorar a logística entre diferentes regiões.

Custos Logísticos: Apresentar gráficos de barras ou mapas com os custos logísticos por região, com a possibilidade de realizar filtros.

Melhores Práticas de Implementação

Uso de Filtros: Sempre permita que o usuário filtre os dados por período, transportadora ou região. Isso ajuda a focar na análise que mais interessa.

Responsividade: O dashboard deve ser funcional tanto em desktop quanto em dispositivos móveis. Teste a responsividade ao alterar o tamanho da janela do navegador.

Atualização de Dados: Certifique-se de que os dados são atualizados periodicamente para refletir as entregas recentes.

Performance: Use técnicas de cache do Streamlit para melhorar a performance ao carregar grandes volumes de dados.

Entrega e Documentação

Código: Enviar o código completo no Google Classroom.

Documentação Técnica: Incluir todas as instruções necessárias para a execução do projeto, incluindo instalação de dependências, configuração do ambiente e execução do código.

Tecnologias Utilizadas: Especificar as bibliotecas e versões do Python utilizadas no projeto.