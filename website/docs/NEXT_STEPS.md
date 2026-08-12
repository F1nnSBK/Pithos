# Pithos – Roadmap & Next Steps (Architektur-Erweiterungen)
*Entwurf: 18. Juni 2026*

Dieses Dokument beschreibt die geplante Roadmap zur Erweiterung der Pithos Vektorsuchmaschine. Ziel ist es, die bestehenden Trade-offs (Genauigkeitsverlust, statischer Index, Overhead bei kleinen Dimensionen) gezielt zu adressieren, ohne die herausragenden Leistungsmerkmale der Engine zu gefährden.

---

## 1. Flexible & konfigurierbare Quantisierungs-Engine

Um die Genauigkeit (Recall) flexibel an die Anforderungen der Anwendung anzupassen, soll die Binarisierung von einem sturen 1-Bit-System in eine wählbare Strategie überführt werden. Der Endnutzer entscheidet bei der Initialisierung/Kompilierung des Index über den Modus:

### 1.1 Quantisierungs-Modi (QuantizationMode)
* **MODE_1BIT (Standard):**
  * *Funktionsweise:* Speichert nur das Vorzeichen ($+1$ / $-1$).
  * *Eignung:* Maximale Geschwindigkeit, extrem geringe Speicherbandbreite. Ideal für Sonden und Edge-Hardware.
* **MODE_2BIT (Ternary / Amplitude-Aware) [UMGESETZT]:**
  * *Status:* Vollständig implementiert und evaluiert (Stand: 23. Juni 2026).
  * *Funktionsweise:* Codiert pro Dimension zwei Bits, um drei Zustände abzubilden: $-1$, $0$ (Wert nahe Null / Rauschen) und $+1$.
  * *Eignung:* Deutlich höherer Recall, da dimensionale Störsignale (Rauschen) durch den Zustand $0$ maskiert werden. Geringfügiger Rechen-Overhead im Vergleich zu 1-Bit.
* **MODE_FLOAT_HYBRID (Raw Bypass):**
  * *Funktionsweise:* Speichert rohe Floats für Dimensionen $D \le 32$ und umgeht die Quantisierung komplett.

### 1.2 API-Entwurf zur Initialisierung (C-Schnittstelle)
Die Funktion zur Index-Kompilierung wird erweitert, um eine Konfiguration zu übergeben:

```c
typedef enum {
    QMODE_1BIT = 0,
    QMODE_2BIT = 1,
    QMODE_FLOAT_HYBRID = 2
} QuantizationMode;

// API zur Index-Kompilierung mit flexibler Quantisierung
int vdb_compile_index_file(
    graal_isolatethead_t* thread, 
    char* path, 
    byte planetId, 
    long long planetRadius, 
    int dimension, 
    int* tiers, 
    int numTiers, 
    long long* ids, 
    float* vectors, 
    int numRecords,
    QuantizationMode qMode
);
```

---

## 2. Zweistufiges In-Engine Reranking (Hybrid Index)

Um bei großen Dimensionen einen Recall von nahe 100% zu erreichen, implementiert Pithos eine integrierte Nachsortierung direkt auf der C-Ebene (off-heap), ohne den Speicherbus-Vorteil zu verlieren.

```mermaid
graph LR
    A[Query Vector] --> B["1-Bit / 2-Bit Scan (Hauptdatenbank)"]
    B --> C["Filtere Top-500 Kandidaten (IDs)"]
    C --> D["Native FP16 L2 Distanzberechnung (in-engine)"]
    D --> E["Lokal sortierte Top-K Treffer"]
```

### Details zur Umsetzung:
1. **Daten-Layout:** Die Index-Datei speichert neben den binären Tiers auch die originalen Vektoren im komprimierten **Float16-Format (FP16)** ab.
2. **Kaskadierter Ablauf:** Der Hamming-Scan filtert blitzschnell die Top $K_{\text{candidate}} = 500$ Kandidaten-IDs heraus.
3. **Local Rerank:** Direkt im Anschluss lädt der native Java-Core off-heap die 500 zugehörigen FP16-Vektoren und berechnet die exakte L2-Distanz. Nur die finalen Top-$K$ (z. B. 10) werden über die FFI-Grenze an Python zurückgegeben.
4. **Ergebnis:** Nahezu perfekter Recall (95%+) bei minimaler I/O-Last (da nur 500 Vektoren hochauflösend geladen werden müssen).

---

## 3. Log-Structured Merge Index (Schreibbarer Delta-Puffer)

Um Echtzeit-Inserts zu ermöglichen, ohne das extrem schnelle, lineare Lese-Layout (`mmap`) zu zerstören, wird ein LSM-ähnlicher Ansatz gewählt:

* **Delta-Puffer (MemTable):** Ein kleiner, beschreibbarer In-Memory-Vektorindex (z. B. ein einfacher Flat-Array oder ein kleiner HNSW-Index), der neue Einfügungen im Millisekunden-Takt aufnimmt.
* **Base-Index (Immutable SSTable):** Der große, schreibgeschützte Hauptindex auf der Festplatte.
* **Unified Search:** Suchanfragen fragen parallel den Hauptindex (Pithos-Kaskade) und den dynamischen Delta-Puffer ab. Die Ergebnisse werden im C-Kernel gemerged.
* **Background Flush:** Sobald der Delta-Puffer eine Grenze (z. B. 10.000 Vektoren) erreicht, wird er im Hintergrund binarisiert und in einer sequenziellen Merge-Operation in die Hauptdatenbank geschrieben.

---

## 4. Dimensions-Adaptive SIMD-Kerne (Vector API)

Für kleinere Dimensionen ($D \le 32$) verzichtet Pithos künftig auf die Bit-Kompression und nutzt stattdessen direkt die Hardware-SIMD-Register der CPU über die neue **Java 25 Vector API**:

* **Auto-Dispatching:** Beim Laden des Indexes wird die Dimension geprüft.
* **Execution Path:**
  * Bei $D \le 32$ wird ein optimierter Floating-Point-L2-Kernel ausgeführt.
  * Bei $D \ge 64$ greift der Matryoshka-Kaskaden-Hamming-Scan.
* Dadurch deckt Pithos alle Dimensionen performant ab und schlägt FAISS auch im niedrigen Dimensionsbereich.
