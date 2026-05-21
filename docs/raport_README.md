# Raport końcowy — `raport_koncowy.tex`

LaTeX-owy raport końcowy projektu **DDDS**, w stylu i strukturze
sprawozdania z laboratorium (na wzór wcześniejszego raportu *waveHome*).

Po skompilowaniu daje **~25-stronicowy** PDF z:

* stroną tytułową w stylu Politechniki Lubelskiej,
* spisem treści,
* streszczeniem ze słowami kluczowymi,
* 19 rozdziałami merytorycznymi (wprowadzenie, cel, analiza problemu,
  założenia, architektura, hardware, software, implementacja z 9 listingami
  kodu, działanie pipeline'u, stabilizacja, koncepcja TDOA z równaniami,
  analiza latencji, UI, testowanie, koszty, bezpieczeństwo, rozwój,
  demonstracja, wnioski),
* bibliografią (25 pozycji).

## Kompilacja --- wariant najprostszy: **Overleaf** (zalecany)

1. Zaloguj się na <https://www.overleaf.com/>.
2. **New Project → Upload Project** → wgraj plik `raport_koncowy.tex`.
3. Po lewej w panelu projektu kliknij **Menu** i upewnij się, że
   compiler ustawiony jest na **pdfLaTeX**.
4. Kliknij **Recompile**. Pierwsza kompilacja zajmuje ~30 sekund (Overleaf
   doinstalowuje brakujące pakiety LaTeX), kolejne ~5 sekund.
5. Pobierz PDF przyciskiem ⬇.

Overleaf nie wymaga niczego instalować lokalnie i ma już wszystkie pakiety
TeX Live, w tym `babel-polish`, `listings`, `booktabs`, `tikz`, `hyperref`.

## Kompilacja --- wariant lokalny (Windows + MiKTeX)

1. Zainstaluj **MiKTeX** z <https://miktex.org/download>.
2. Podczas pierwszej kompilacji MiKTeX automatycznie dociągnie brakujące
   pakiety; potwierdź **Install** dla każdego.
3. W PowerShell:

   ```powershell
   cd C:\Users\cebularz\drone-detector\docs
   pdflatex raport_koncowy.tex
   pdflatex raport_koncowy.tex   # drugi run dla poprawnego ToC i linków
   ```

   Wynik: `raport_koncowy.pdf` w tym samym katalogu.

## Kompilacja --- wariant Linux / WSL

```bash
sudo apt install texlive-full
cd ~/drone-detector/docs
pdflatex raport_koncowy.tex
pdflatex raport_koncowy.tex
```

(`texlive-full` to ok. 4 GB; jeśli zależy ci na rozmiarze, wystarczy
`texlive-latex-extra texlive-lang-polish texlive-fonts-extra texlive-pictures`.)

## Co zmienić w pliku przed obroną?

* **Strona tytułowa** (linie 90--130) --- zaktualizuj rok, datę oddania,
  adres GitHub jeśli zmieniłeś nazwę repo.
* **Rozdział 18 Demonstracja** --- jeśli nagrasz wideo demonstracyjne,
  zamień placeholder `docs/images/demo.mp4` na realny link YouTube
  (analogicznie do tego, jak `waveHome` ma link `youtube.com/shorts/...`).
* **Rozdział 14 Testowanie** --- liczby `91%`, `0.96 AUC` itd. są
  *placeholderami* przykładowymi. **Zastąp je realnymi liczbami** po
  uruchomieniu `drone-detector train` na własnym zbiorze (zostaną w
  `reports/evaluation.md`).
* **Bibliografia** --- pozycje typu `arXiv:2403.XXXXX` to placeholdery.
  Wstaw realne odnośniki lub usuń.
* **Logo / screeny** --- możesz dodać do katalogu `docs/images/`:
  - screenshot strony *Live microphone* z bannerem DETECT,
  - screenshot strony *4-mic TDOA concept* z geometrią,
  - confusion matrix wygenerowaną przez `drone-detector train`.

  Następnie w pliku `.tex` dodaj `\includegraphics{...}` w odpowiednich
  miejscach (możesz wzorować się na sposobie, w jaki `waveHome` ma
  `Rysunek 2` / `Rysunek 3`).

## Struktura logiczna raportu

```
Streszczenie + słowa kluczowe
├── 1.  Wprowadzenie                       (kontekst — Shahed-136, Ukraina, Polska)
├── 2.  Cel i zakres projektu              (cele główne + szczegółowe + poza zakresem)
├── 3.  Analiza problemu                   (tabela porównawcza istniejących rozwiązań)
├── 4.  Założenia systemowe                (WF + WN)
├── 5.  Architektura systemu               (warstwowa + diagram TikZ)
├── 6.  Architektura sprzętowa             (tabela komponentów)
├── 7.  Architektura programowa            (tabela tech-stacku + struktura repo)
├── 8.  Implementacja                      (9 listingów kodu z uzasadnieniem)
├── 9.  Działanie pipeline'u MFCC + RF     (krok po kroku + dlaczego)
├── 10. Stabilizacja i progowanie
├── 11. Koncepcja 4-mikrofonowego TDOA     (równania + tabela rozdzielczości)
├── 12. Analiza latencji                   (tabela dekompozycji)
├── 13. Wizualizacja i UI                  (opis 5 stron Streamlit)
├── 14. Testowanie systemu                 (środowisko + testy + metryki + odporność)
├── 15. Analiza kosztów                    (BOM faza obecna + docelowa)
├── 16. Bezpieczeństwo, prywatność, etyka  (privacy by design + granica defensywna)
├── 17. Możliwości rozwoju                 (9 punktów po trudności rosnącej)
├── 18. Demonstracja działania             (scenariusz obronny, ~8 min)
├── 19. Wnioski końcowe                    (6 kluczowych obserwacji technicznych)
└── Bibliografia                           (25 pozycji)
```

Sumaryczna objętość po skompilowaniu: 22--26 stron PDF (zależnie od długości
list literatury i ewentualnych wstawionych obrazków).
