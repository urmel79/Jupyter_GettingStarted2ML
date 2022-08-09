# Getting started with Machine Learning (ML) and Support Vector Classifiers (SVC) - A systematic step-by-step approach

## Abstract (de/en)
Wer sich mit dem Hypethema unserer Zeit ``Künstliche Intelligenz (KI)'' bzw. ``Machine Learning (ML)'' ernsthaft auseinandersetzen möchte, kommt nicht umhin, sich mit den grundlegenden ML-Algorithmen, entsprechenden Software-Werkzeugen, -Bibliotheken und Programmiersystemen zu beschäftigen.

Anyone who wants to seriously deal with the hypothetical topic of our time ``Artificial Intelligence (AI)'' or ``Machine Learning (ML)'' cannot avoid dealing with the basic ML algorithms, corresponding software tools, libraries and programming systems.

Allerdings wird jemand, der das erste Mal die Tür zu dieser gleichermaßen sehr spannenden wie beliebig komplexen und auf den ersten Blick unübersichtlichen Welt aufstößt, sehr schnell überfordert sein. Hier bietet es sich an, einführende und systematische Anleitungen zu konsultieren.

However, someone who opens the door for the first time to this equally very exciting as well as arbitrarily complex and, at first glance, confusing world will very quickly be overwhelmed. Here, it is a good idea to consult introductory and systematic tutorials.

Daher demonstriert das vorliegende Getting-Started-Tutorial anhand des sehr leistungsfähigen und performanten ``Support Vector Classifiers (SVC)'' sowie dem weithin bekannten und besonders anfängerfreundlichen ``Iris-Datensatz'' den typischen ML-Arbeitsprozess systematisch Schritt-für-Schritt.

Therefore, this Getting Started tutorial systematically demonstrates the typical ML work process step-by-step using the very powerful and performant ``Support Vector Classifier (SVC)'' and the widely known and exceptionally beginner-friendly ``Iris Dataset''.

Darüber hinaus werden die Auswahl des ``richtigen'' SVC-Kernels sowie deren Parameter beschrieben und ihre Auswirkungen auf das Klassifikationsergebnis werden gezeigt.

Furthermore, the selection of the ``correct'' SVC kernel and its parameters are described and their effects on the classification result are shown.

## Introduction (de/en)

Von den **Arbeitsmitteln** in der **digitalisierten Arbeitswelt** wird immer stärker gefordert, dass sie sich selbstständig und aufgabenbezogen an sich ändernde Arbeitssituationen anpassen können. Diese **situative Adaptivität** kann je nach Stärke des Flexibilisierungsgrades oft nur durch Anwendung von **Artificial Intelligence (AI)** bzw. **Machine Learning (ML)** realisiert werden.

In the **digitised work environment**, there is an increasing demand for **Work equipment** to be able to adapt independently and in a task-related manner to changing work situations. This **situational adaptivity** can often only be realised through the use of **Artificial Intelligence (AI)** or **Machine Learning (ML)**, depending on the degree of flexibility.

Als Beispiele für solche KI-Anwendungen in der Arbeitswelt können vergleichsweise einfache **Sprachassistenzsysteme** (ähnlich z. B. Siri oder Alexa aus dem privaten Umfeld) bis hin zu teil- oder gar **vollautonomen Systemen** genannt werden. Solche vollautonomen Systeme sind beispielsweise sog. **fahrerlose Transportsysteme**, bei denen es sich um autonom fahrende Logistikfahrzeuge in größeren Industrieanlagen handelt.

Examples of such AI applications in work environments can range from comparatively simple **voice assistance systems** (similar, for example, to Siri or Alexa from the private sphere) to partially or even **fully autonomous systems**. Such fully autonomous systems are, for example, so-called **driverless transport systems**, which are autonomously driving logistics vehicles in larger industrial plants.

Neben den vielen sehr interessanten Vorteilen bzgl. Wirtschaftlichkeit, Arbeitserleichterung usw. kennzeichnet solche vollautonomen Systeme eine sehr hohe technische Komplexität. Diese betrifft sowohl ihre **Betriebsfunktionen** (z. B. autonome Navigation durch komplexe industrielle Umgebungen bei gemeinsamer Nutzung der Fahrwege durch andere menschlich gesteuerte Fahrzeuge) als auch seiner **Sicherheitsfunktionen** (z. B. Auswertung miteinander verknüpfter bildgebender mit nicht-bildgebender Sicherheitssensorik zur Überwachung des Fahrraums zur Kollisionsvermeidung).

In addition to the numerous very interesting advantages in terms of economic efficiency, workload reduction, etc., such fully autonomous systems are characterised by a very high level of technical complexity. This concerns both their **operating functions** (e.g. autonomous navigation through complex industrial environments with shared use of the roadways by other human-controlled vehicles) and their **safety functions** (e.g. evaluation of interlinked imaging with non-imaging safety sensors for monitoring the driving space to avoid collisions).

An solche autonomen Systeme und die hierfür eingesetzten KI-Algorithmen werden sehr hohe Anforderungen hinsichtlich der **funktionalen Sicherheit** gestellt. Jedoch sind die Anforderderungen für eine sicherheitstechnische Bewertbarkeit hinsichtlich der **Transparenz** und **Erklärbarkeit** der durch KI getroffenen Entscheidungen je nach verwendeten KI-Algorithmen sehr schwer bis unmöglich erreichbar. Beispielsweise werden durch aktuell laufende Forschungsprojekte die Transparenz und Erklärbarkeit von **tiefen neuronalen Netzen** untersucht. Weiterhin erfüllen heutige KI-Algorithmen hinsichtlich ihrer **Erkennnungsraten** und damit ihrer **Zuverlässigkeiten** selbst unter günstigsten Bedingungen sehr oft nicht die Anforderderungen an die funktionale Sicherheit, um höhere Safety-Level (z. B. Performance Level d (PLd) nach ISO 13849) zu erreichen.

Very high requirements are placed on such autonomous systems and the AI algorithms used for this purpose with regard to **functional safety**. However, the requirements for safety evaluability in terms of **transparency** and **explainability** of decisions made by AI are very difficult or impossible to meet, depending on the AI algorithms applied. For example, current research projects are investigating the transparency and explainability of **deep neural networks**. Furthermore, today's AI algorithms, in terms of their **recognition rates** and thus their **reliabilities**, very often do not meet the functional safety requirements to achieve higher safety levels (e.g. Performance Level d (PLd) according to ISO 13849), even under the most convenient conditions.

Eine hinsichtlich der geforderten funktionalen Sicherheit angemessene Bewertung oder gar **Prüfung** nach einheitlichen und idealerweise genormten Maßstäben hat viele Implikationen auf die zukünftige Ausrichtung des **technischen Arbeitsschutzes** in Deutschland und in Europa. Neben der derzeit noch sehr schwierigen sicherheitstechnischen Bewertbarkeit von KI-Algorithmen ist ein wichtiger Punkt, dass die bisherige klare Trennung zwischen **Inverkehrbringensrecht** (siehe z. B. Maschinenrichtlinie) und **betrieblichem Arbeitsschutzrecht** (siehe Arbeitsschutz-Rahmenrichtlinie und Betriebssicherheitsverordnung) so nicht mehr aufrechterhalten werden kann. Grund hierfür ist, dass sich auch die **sicherheitsrelevanten Eigenschaften** der autonomen Systeme durch während des Betriebs erlernte, neue oder **angepasste Verhaltensweisen** verändern werden.

An appropriate assessment or even **testing** with regard to the required functional safety according to uniform and ideally standardised criteria has many implications for the future orientation of technical **occupational safety and health (OSH)** in Germany and in Europe. In addition to the currently still very difficult safety-related assessability, an important point is that the previous clear separation between **placing on the market law** (see e.g. Machinery Directive) and **occupational safety and health law** (see European Framework Directive for Occupational Safety and Health and German Ordinance on Occupational Safety and Health) can no longer be continued in this way. The reason for this is that the **safety-relevant properties** of the autonomous systems will change due to new or **adapted behaviours** learned during operation.

Aus diesen Gründen sollten sich insbesondere die Akteure des technischen Arbeitsschutzes, die sich zukünftig mit der Prüfung solcher lernfähigen, autonomen Systeme oder Systemkomponenten mit KI-Algorithmen befassen werden, möglichst frühzeitig mit den KI- bzw. ML-Algorithmen vertieft auseinandersetzen. Nur dadurch lässt sich erreichen, dass die stürmische Entwicklung lernfähiger, adaptiver Systeme durch den Arbeitsschutz und deren Prüfinstitute konstruktiv, kritisch und fachlich angemessen begleitet werden kann. Wird dies versäumt, muss aufgrund der Erfahrungen der vergangenen Jahre davon ausgegangen werden, dass das Arbeitsschutzsystem durch die wirtschaftlichen Interessen global agierender Softwaregiganten skrupellos umgangen oder ausgehebelt werden wird. Dies hätte die Folge, dass schwere oder tödliche Arbeitsunfälle wegen unzulänglich gestalteter KI-basierter Arbeitssysteme wahrscheinlich werden.

For these reasons, especially the actors of technical occupational safety and health who will deal with the evaluation of such adaptive, autonomous systems or system components with AI algorithms in the future should familiarize themselves with the AI or ML algorithms in depth as early as possible. This is the only way to ensure that the rapid development of adaptive systems capable of learning can be accompanied by OSH and their testing authorities in a constructive, critical and technically appropriate manner. If this is omitted, it must be assumed on the basis of the experience of recent years that the OSH system will be ruthlessly circumvented or undermined by the economic interests of globally operating software giants. This would have the consequence that serious or fatal occupational accidents are likely to occur due to inadequately designed AI-based work systems.

Allerdings erfordert die sicherheitstechnische Bewertung solcher lernfähigen Systeme einen tiefergehenden fachlichen Einstieg in die Welt von **Künstlicher Intelligenz (KI)** bzw. **Machine Learning (ML)**. Hierzu muss sich mit den grundlegenden Funktionsweise typischer ML-Algorithmen, entsprechenden Software-Werkzeugen, Bibliotheken und Programmiersystemen auseinander gesetzt werden.

However, the safety-related evaluation of such learning-capable systems requires a deeper technical entry into the world of **Artificial Intelligence (AI)** or **Machine Learning (ML)**. For this purpose, it is necessary to deal with the basic operation of typical ML algorithms, corresponding software tools, libraries and programming systems.

Wer jedoch zum ersten Mal die Tür zu dieser ebenso spannenden wie beliebig komplexen und auf den ersten Blick verwirrenden Welt öffnet, wird sehr schnell überfordert sein. Hier empfiehlt es sich neben dem Lesen allgemeiner Fachliteratur, einführende und systematische Anleitungen zu Rate zu ziehen.

However, someone who opens the door for the first time to this equally very exciting as well as arbitrarily complex and, at first glance, confusing world will very quickly be overwhelmed. In addition to reading general technical literature, it is advisable to consult introductory and systematic tutorials.

Genau dieses Ziel verfolgt das vorliegende Getting-Started-Tutorial, indem systematisch und Schritt-für-Schritt der typische ML-Arbeitsablauf am Beispiel des sehr leistungsfähigen **Support Vector Classifier (SVC)** demonstriert wird.

This Getting Started tutorial has exactly this goal, demonstrating systematically and step-by-step the typical ML workflow using the very powerful **Support Vector Classifier (SVC)** as an example.

Dieses Tutorial wird im Rahmen eines Workshops auf der DGUV-Fachtagung **Künstliche Intelligenz** voraussichtlich im November 2022 in Dresden vorgestellt. Der Workshop richtet sich an interessierte ML-Neulinge im technischen Arbeitsschutz der gesetzlichen Unfallversicherungsträger.

This tutorial will be presented as part of a workshop at the DGUV symposium **Artificial Intelligence**, probably in November 2022 in Dresden. The workshop addresses interested ML novices in the technical occupational safety and health of the social accident insurance institutions.

Neben den medial sehr präsenten **tiefen neuronalen Netzen** gibt es eine sehr reichhaltige Auswahl anderer sehr leistungsfähiger ML-Algorithmen - passend für den jeweiligen Anwendungsfall. Für einen allgemein verständlicheren Einstieg wurde für die Zielgruppe des Workshops der SVC-Algorithmus bewusst gewählt. Dessen Arbeitsweise ist sowohl für ML-Neulinge als auch in dem für den Workshop vorgegebenen Zeitrahmen leicht vermittelbar - ganz im Gegensatz zum Einstieg in die Welt der tiefen neuronalen Netze.

Besides the **deep neural networks**, which are very present in the media, there is a very rich selection of other very powerful ML algorithms - suitable for the particular use case. For a more generally comprehensible introduction, the SVC algorithm was deliberately chosen for the target audience of the workshop. Its operating principles are easy to convey to ML novices as well as in the time frame given for the workshop - quite in contrast to the entry into the world of deep neural networks.

Die folgenden Hauptabschnitte demonstrieren den typischen ML-Arbeitsablauf Schritt-für-Schritt. Im **Schritt 0** werden konkrete Hinweise für die Auswahl der für das maschinelle Lernen geeignete Hardware und Software gegeben. Damit sich ein ML-Neuling zunächst mit den ML-Algorithmen, Werkzeugen, Bibliotheken und Programmiersystemen vertraut machen kann, wird im **Schritt 1** ein fertiger und ML-tauglicher Datensatz hinzugezogen. Erst danach wäre es sinnvoll, die eigene Umgebung auf ML-taugliche Anwendungen hin zu untersuchen und daraus geeignete Datensätze zu gewinnen. Dies geht jedoch über den Rahmen dieses einführenden Tutorials hinaus.

The following main sections demonstrate the typical ML workflow step-by-step. In **Step 0**, specific guidance is provided for selecting hardware and software suitable for machine learning. To first familiarize a novice ML user with ML algorithms, tools, libraries, and programming systems, a ready-made and ML-suitable dataset is involved in **Step 1**. Only then would it make sense to examine one's own environment for ML-suitable applications and to acquire appropriate datasets from them. However, this is beyond the scope of this introductory tutorial.

Mit am wichtigsten im gesamten ML-Prozess ist **Schritt 2**: der in Schritt 1 bezogene sehr einsteigerfreundliche **Iris-Datensatz** wird mit Hilfe typischer Datenanalyse-Werkzeuge untersucht. Neben der Erkundung innerer Zusammenhänge im Datensatz müssen auch Fehler wie z. B. Lücken, Dopplungen oder offensichtliche Fehleingaben gefunden und nach Möglichkeit behoben werden. Dies ist enorm wichtig, damit die Klassifikation später plausible Ergebnisse liefern kann.

Among the most important in the entire ML process is **step 2**: the extremely beginner-friendly **Iris dataset** obtained in step 1 is examined using typical data analysis tools. In addition to exploring internal correlations in the dataset, errors such as gaps, duplications, or obvious misentries must also be found and corrected where possible. This is enormously important so that the classification can later provide plausible results.

Im **Schritt 3** wird der Datensatz für die eigentliche Klassifikation per SVC im **Schritt 4** vorbereitet. Neben anderen möglichen ML-Algorithmen (z. B. der entscheidungsbaum-basierte **Random-forests-Klassifikator**) ist laut Fachliteratur der **Support-Vector-Klassifikator** für die Klassifikation des Iris-Datensatzes hinsichtlich Erkennungsrate als auch Performanz besonders gut geeignet.

In **step 3**, the dataset is prepared for the actual classification by SVC in **step 4**. Among other possible ML algorithms (e.g., the decision tree-based **random-forests classifier**), the **support vector classifier** is particularly well suited for the classification of the iris dataset in terms of recognition rate as well as performance, according to the literature.

Die Güte des Klassifikationsergebnisses wird im **Schritt 5** anhand bekannter **Metriken** evaluiert. Da die Klassifikation im Schritt 4 zunächst mit Standard-Parametern (sog. **Hyper-Parameter**) durchgeführt wurde, werden diese im **Schritt 6** zunächst erklärt und danach ihr Einfluss auf das Klassifikationsergebnis durch Variation der einzelnen Hyper-Parameter demonstriert.

The quality of the classification result is evaluated in **step 5** using known **metrics**. Since the classification in step 4 was initially performed with standard parameters (so-called **hyper-parameters**), these are first explained in **step 6** and then their effect on the classification result is demonstrated by varying the individual hyper-parameters.

Im abschließenden **Schritt 7** werden zwei Ansätze zur systematischen Hyper-Parameter-Suche vorgestellt: **Grid Search** und **Randomized Search**. Während erstere für gegebene Werte erschöpfend alle Parameterkombinationen betrachtet, wählt der zweite Ansatz eine bestimmte Anzahl von Kandidaten aus einem Parameterraum mit einer bestimmten zufälligen Verteilung aus.

In the final **step 7**, two approaches for systematic hyper-parameter search are presented: **Grid Search** and **Randomized Search**. While the former exhaustively considers all parameter combinations for given values, the latter approach selects a certain number of candidates from a parameter space with a particular random distribution.

## Steps of the systematic ML process

+++
@TODO: Adapt section headers and internal links.
+++

The following **steps of the systematic ML process** are covered in the next main sections:

- [STEP 0: Select hardware and software suitable for ML](#STEP-0:-Select-hardware-and-software-suitable-for-ML)
- [STEP 1: Get the ML dataset](#STEP-1:-Get-the-ML-dataset)
- [STEP 2: Explore the ML dataset](#STEP-2:-Explore-the-ML-dataset)
- [STEP 3: Prepare the dataset for training](#STEP-3:-Prepare-the-dataset-for-training)
- [STEP 4: Classify by support vector classifier - SVC](#STEP-4:-Classify-by-support-vector-classifier---SVC)
- [STEP 5: Evaluate the classification results by metrics](#STEP-5:-Evaluate-the-classification-results-by-metrics)
- [STEP 6: Select SVC kernel and vary parameters](#STEP-6:-Select-SVC-kernel-and-vary-parameters)
- [STEP 7: Search for SVC parameters systematically](#STEP-7:-Search-for-SVC-parameters-systematically)


