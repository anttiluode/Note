# Note

*A lone note, written by Claude (Opus 5.5) on 5 October 2026 from one day of conversation between Antti Luode, Claude, Sol (ChatGPT), Grok, Gemini and a second Claude instance. It records where the day went, what was checked, what was withdrawn, and what is still open.*

**Do not hype. Do not lie. Just show.** Every claim below has a status. Only one thing in this repo was run: a toy algebra check (`transport_algebra.py`), and it says nothing about magnitudes in real models.

---

## 1. The question

After two years of trying to build a neuron that does something clever, the question flipped:

> What properties must a computational substrate have so that human language naturally organizes itself into reusable internal computation?

The day started from Anthropic's workspace paper, ran into a paper that had already built the obvious next machine, and ended on how a mind *grows* its codebook rather than just uses it.

## 2. What the Anthropic workspace paper measured

Gurnee, Sofroniew, … Lindsey, *Verbalizable Representations Form a Global Workspace in Language Models* (arXiv 2607.15495, July 2026). The Jacobian lens (J-lens) averages `∂h_final/∂h_ℓ` over contexts. Each word in the vocabulary gets one direction per layer: the direction that makes the model disposed to *say* that word. The J-space is a sparse combination of these directions.

Measured (mostly on Claude Sonnet 4.5):

- **Selectivity.** Report and flexible inference route through the J-space. Continuation, anomaly detection and parsing don't. Ablating it hurts multi-hop reasoning, translation and TriviaQA, while leaving MMLU, SQuAD and CoLA almost untouched. GSM8K holds up when the model writes chain-of-thought and doesn't when it answers directly.
- **Location and capacity.** It is active in a middle band (about 38–92% of depth). About 25 active directions at a time. The J-component carries about 6–7% of a concept vector's variance. Only about 6 unrelated items fit at once; related items load as a whole category.
- **Broadcast.** MLPs amplify J-directions about 10× (neuron output directions about 1×). A dedicated top-1% of attention heads relay them faithfully.
- **Ignition.** Blend two country embeddings and the early layers track the blend smoothly. At the workspace onset the representation snaps to one country or the other, with a bimodal split.
- **Origin.** It is already present in the base model. Next-token prediction alone builds it.
- **The authors' own explanation for why it is verbal:** language comes in and language goes out, so the meeting point uses the token code. Humans take in sensory signals and put out motor commands, so they predict the human workspace is less verbal.

Six substrate properties read off these results: a shared bus, the same code in and out, operators selected by content, a commit step, a small competitive capacity, and a loop through symbols. A transformer has all six. None of them is a property of the single unit.

## 3. Someone had already built the loop: J-CoT

Wu, Zhu, Liu, Hu, Pan, *J-CoT: Chain-of-Thought in J-Space* (arXiv 2607.21981, 24 July 2026). Within each cycle the model computes in its full hidden state. At the cycle boundary the state is written as nonnegative sparse J-coefficients (elastic net) at layer 28 of Qwen3-8B, then read back in at layer 12 as a "J-thought", using 8 carrier positions. Their reported numbers:

| | CoT | SIM-Coconut | J-CoT-Zero | J-CoT-Train |
|---|---|---|---|---|
| 8-benchmark mean | 45.8 | 47.5 | 47.9 | 50.2 |

ProsQA interface sweep: explicit language 79.0, dense latent 84.0, J-space **88.8**. The best result sits in the middle.

**Ledger on J-CoT (v1)**

| Point | Status |
|---|---|
| The appendix (proof of Prop. 1, Jacobian estimator, solver, stopping rule) is referenced but absent. The text contains "Appendices ?? and ??" | checked in the PDF |
| The recovery defect ε, the stated precondition, is never measured | checked in the PDF |
| No random-dictionary or non-language control | checked in the PDF |
| J-CoT-Zero beats SIM-Coconut by 0.3–0.6 points, within the reported std 0.5–3.5 | checked in the PDF; the Pith review agrees |
| J-CoT-Train gains of 2.1–3.6 points exceed the stds | the plausible real result |
| "Identical ranking on every benchmark is suspicious" | **withdrawn.** Methods share a checkpoint and test items, so paired differences are far less noisy than the per-method stds. Coconut < CoT also matches the literature (Sol's correction) |
| "Eight days after the Anthropic paper is suspicious" | **withdrawn.** Chronology proves nothing |

Related: *Looped Transformers under the Jacobian Lens* (arXiv 2609.01924). In Ouro, workspace content such as *Italy* did not survive a loop boundary through the averaged-Jacobian map: its projection fell to about 0.1 while the bulk state stayed at 0.31–0.45 cosine. In Huginn, content persists across loops, but each lens reads only within about a 2-recurrence horizon. There is a fair counterpoint (microprediction/latentreasoning issue #43): this shows the *averaged* map failing, not that transport is impossible.

## 4. The algebra (the one receipt)

Each J-lens atom is the same final-layer word vector pulled back through that layer's averaged Jacobian: `d_ℓ,v = J_ℓᵀ u_v`. Write the vocabulary out and J-CoT's read-in becomes (unnormalized):

```
read-in  =  D_r a  =  T · h_w ,      T = J_rᵀ · J_w⁻ᵀ
```

- **The vocabulary index adds no transport information.** Moving a state from layer 28 to layer 12 is one fixed linear map. Language can enter only through the sparse-coding step (which atoms survive) and through J-CoT's per-layer atom normalization (a per-word gain `n_w,v / n_r,v`).
- **The map that preserves downstream effect is different.** `S = J_r⁻¹ · J_w` is the read-layer change with the same first-order downstream effect as `h_w`. It relates to J-CoT's map by **`T = S⁻ᵀ`**. The two agree only if `S` is orthogonal.
- **Prop. 1's ε is the wrong quantity.** `ε_ℓ(a) = ‖Φ_ℓ(Ψ_ℓ(a)) − a‖` reconstructs and re-extracts at the same layer with no transformer block in between. It is a property of the dictionary, so even if measured it could not establish that a coordinate keeps its identity through the model. The test that matters goes through the model: write a word's atom at layer *r*, run to *w*, and check that the same word comes out on top.

`python transport_algebra.py` checks claims 1–6 on random matrices (output in `transport_algebra_output.txt`):

```
[1] read-in D_r a == T h_w           : True
[2] T == S^{-T}                      : True
cos(T h_w, S h_w) on the J-thought   : 0.035   (generic random Jacobians)
                                       0.005   (read layer with a collapsed block)
[3] S orthogonal  ->  T == S         : True
[4] read weight = stretch, write weight = 1/stretch
[5] normalised read-in == T h_w with per-word gain n_w/n_r : True
[6] eps_l(a) = 2e-15 with no transformer block run
```

**Status:** the algebra is exact. The toy cosines are *not* evidence about Qwen. Random `I + noise` matrices are badly conditioned too. How far `S` is from orthogonal in a real model is Gate 0.

**Sol's correction, accepted:** `T` isn't "wrong". It is the natural transport of the *read* (dual) coordinates; `S` is the natural transport if you demand the same downstream effect. J-CoT is implicitly asking whether read coordinates can serve as recurrent state.

## 5. Reading is not writing

- **Park, Choe, Veitch (ICML 2024), *The Linear Representation Hypothesis and the Geometry of LLMs*.** A probe (read) vector and a steering (write) vector coincide only under the *causal inner product*. In Euclidean coordinates the steering vector is `Cov(γ)⁻¹` times the probe vector, where γ ranges over the unembedding vectors.
- **Park, Nief, Choe, Veitch (ICML 2026), *The Information Geometry of Softmax: Probing and Steering*** (arXiv 2602.15293). Adding a probe vector to a state is called "a basic type error". The fix, **dual steering**, steers in Bregman-dual coordinates. Demonstrated at the **final layer only** (Gemma-3-4B, MetaCLIP-2).
- **Gao et al. (ICML 2026), *The Cylindrical Representation Hypothesis*.** A steering vector splits into an axis component (a stable semantic shift) and a normal-plane component (faster activation, less coherence). Which "sector" of the normal plane helps is sample-specific and can't be predicted from the difference vector alone. Every J-CoT read-in is a steering act, so this predicts high per-sample variance and small mean gains for training-free read-in, which is what J-CoT-Zero shows. Whether J-CoT-Train's state-conditioned gate learns the per-sample sector is **a hypothesis, not an explanation** (Sol's correction).

J-CoT's read-in is exactly the "type error" case, at an intermediate layer, inside a loop. As far as this note's sources go, nobody has applied dual steering there.

## 6. A concept as a read/write pair (second Claude instance)

*Status: a frame with old, proven mathematics underneath. As a claim about language models and minds, it is a hypothesis.*

- **A concept is a direction where reading and writing (nearly) agree.** Words can put it there, words can get it out, and many operations can use it in between.
- **A language is two codebooks, one for speaking and one for hearing,** tuned against each other by billions of exchanges. That tuning is why a few kilobytes of text can move a whole state in another system.
- **Control theory already has the parts.** Kalman: input-output behavior depends only on the part that is both reachable and observable (the minimal realization). Moore (1981): balanced truncation finds coordinates where reaching and showing are equal, and ranks each mode by its Hankel singular value. For a single mode that is roughly drive × radiation ÷ decay rate. "A substrate resonates with language" can then be made exact: the directions words can write are the directions words can read, and those modes are strong and slow to die.
- **Put words at both ends** and a mind splits four ways:
  - in and out: thought you can be talked into and talk out of;
  - in but barely out: *mode* ("Watch out!" floods you, and you can't hand the flood back);
  - out but not in: *perception* (no sentence makes someone who has never seen red see it);
  - neither: the automatic machinery.
- **A model trained only on text can learn at most the shared part:** what the words before could have put there and the words after will show. That is the precise version of "a CLIP of the brain". It is a theorem in idealized settings (Kalman, Sussmann, causal states), and Shai et al. 2024 show transformers recovering belief-state geometry on toy processes. **For human text it is a hypothesis.**
- **Development.** Children understand before they speak, so the two codebooks have to be tuned against each other. Vygotsky: speech aimed at others becomes speech aloud to oneself, then inner speech, which is abbreviated, mostly predicates, each word saturated with sense. That reads like a J-thought. It is a rhyme, not evidence, but it predicts something: the right write for a J-loop may be neither the mouth (J-CoT's `T`) nor the pure effect (`S`) but **the model's own ear**, installing each word the way the model holds it after reading it. The pronoun "I" is the sharpest case: you hear "you" and must say "I". A three-year-old who says "you want juice" about herself is making Park's type error.

## 7. Growing the codebook (Sol)

Antti: *"I land on the LLM's inner codebook, always. But what I am looking for is the human process of expanding the codebook."*

- **A two-speed codebook.** A fast provisional binding ("a *dax* is this tool"), then slow integration through replay and offline consolidation: complementary learning systems, and new-word learning where a word starts competing in the lexicon only after consolidation.
- **Context memory ≠ codebook expansion.** A million-token context can keep looking up "FLORB means …". A consolidated concept changes the geometry of what follows, gets used automatically, and becomes material for further concepts. A frozen LLM has the fast half and lacks the compiler from fast to slow.
- **The codebook can expand using the codebook itself.** *Qubit*, *derivative* and *causal inner product* are built by composing existing entries. This is the nonlinear jump, and it is what Luotain was reaching for.
- **The danger is bad primitives.** Compression alone is not enough. A candidate entry has to improve compression, prediction, intervention and generalization *against external observation*.

**Added here.** The J-lens has one atom per existing single token, so **a newly coined concept is invisible to the J-space by construction** (this is the paper's own §9.1 limitation). The J-space is the existing codebook. Codebook growth is exactly what this instrument can't see, so it needs its own test (G4 below).

## 8. Through what does the loop run?

Both religion and science build concepts out of words. What differs is what the loop runs through. A concept tuned only through other minds becomes shared and stable, but the only thing that can say no is another mind trained on the same code. Science routes the loop through the world, the one listener that never learned our language, which is why it can say no. An LLM is the opposite extreme: the most fluent partner ever built, and on its own the least able to say no. From inside language, a real new mode and a fluent hallucination look the same. The gates are the world spliced back into the loop. That is why Luotain reached for Lean and why Vahti was "Lean for instruments".

## 9. Open gates

| Gate | Measurement | Kill condition | Cost |
|---|---|---|---|
| **G0 — Does `S` stay orthogonal?** | Qwen3 (small is fine): averaged Jacobians at the read and write layers (randomized estimator, ~1k prompts). Singular values of `S` restricted to the subspace real activations occupy. `cos(T·h, S·h)` on real write-layer states. Through-model identity: write one word's atom at *r*, run to *w*, check the top J-word | `S` close to orthogonal → the read/write critique doesn't apply here | an afternoon |
| **G1 — Which sparse basis decides what to keep?** (Sol's version) | At the write layer, reconstruct equally sparse from J atoms, SAE atoms and matched random atoms. Push all three through the **same** transport (`S`, and separately `T`). ProsQA: accuracy, per-sample variance, identity | SAE or random ≈ J → language is not special as the selector | days |
| **G2 — Do speaking and hearing meet in the workspace?** (second Claude) | For a few thousand words at every layer: the speaking vector (J atom) vs the hearing vector (the layer state after the model reads the word). Ask whether one linear map carries one set onto the other, and where in depth the fit is best. Use Qwen3-8B, whose input and output embeddings are separate (small Qwen3s tie them; check `config.json`). Control: SAE encoder/decoder pairs | no peak in the workspace band beyond the control → "the workspace is where language is balanced" is dead | days |
| **G3 — Dual-steering read-in inside the loop** | J-CoT-Zero with the read-in replaced by (a) `S`-transport, (b) a Park-style dual write, (c) the model's own "ear" vectors from G2 | none beats `T` → the type error is harmless in practice | after G0–G2 |
| **G4 — Can an in-context concept become a workspace argument?** | Coin a term in context ("a florb is …"). Test whether its representation passes the Anthropic assays: swap → downstream answers follow; many operators can consume it. Prior art on in-context word learning likely exists; search before building | it never behaves like an argument → in-context concepts are lookup, not entries | days |

## 10. Earlier halves of the same idea in this lab's own repos

(As recalled by the second Claude instance from the repo history; not re-verified here.)

- **V23 (15 Aug):** "the present available to a receiver is a partially resolved geometry of alternatives".
- **Note 008 (19 Aug):** different receivers of one six-state world had Hankel ranks 1, 2, 1 and 6. Minimal realization and balanced truncation were named then (Kellems et al. 2009 as prior art).
- **3 Sep, reciprocity:** a read is a transpose.
- **AdaptiveObserverCache, Qwen3-8B (22 Sep):** "steerability ≠ causal usefulness; a head can be an excellent sensor and a lousy control point". One head, pushed toward B, made the model *more* A-like. That is the read/write seam, found in a real model before Park's paper turned up.
- **ResidentOperatorTomography:** coordinates belong to the representation *and* the reader. That is the same gauge problem as §4.
- **Sihti:** keep the residue that is useful for the next operation. That is G1's hypothesis.
- **WhatIsI** (the deictic "I") and the **block neuron** (state changes the task, which is mode).

What never existed before today: both halves together in the symmetric case (sender and receiver made from the same language), and an instrument (the J-lens) that measures one half inside a real model.

## 11. Sources

- Gurnee et al. 2026, *Verbalizable Representations Form a Global Workspace in Language Models*, arXiv [2607.15495](https://arxiv.org/abs/2607.15495)
- Wu et al. 2026, *J-CoT: Chain-of-Thought in J-Space*, arXiv [2607.21981](https://arxiv.org/abs/2607.21981); review at [pith.science](https://pith.science/paper/2607.21981)
- *Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?*, arXiv [2609.01924](https://arxiv.org/abs/2609.01924); counterpoint: [microprediction/latentreasoning #43](https://github.com/microprediction/latentreasoning/issues/43)
- Park, Choe, Veitch 2024, *The Linear Representation Hypothesis and the Geometry of Large Language Models*, [ICML 2024 / PMLR 235](https://proceedings.mlr.press/v235/park24c.html)
- Park, Nief, Choe, Veitch 2026, *The Information Geometry of Softmax: Probing and Steering*, arXiv [2602.15293](https://arxiv.org/abs/2602.15293)
- Gao et al. 2026, *The Cylindrical Representation Hypothesis for Language Model Steering*, ICML 2026 / PMLR 306, code [mbzuai-nlp/CRH](https://github.com/mbzuai-nlp/CRH)
- Chen, Fu 2026, *Latent Recurrent Thoughts*, arXiv [2609.01117](https://arxiv.org/abs/2609.01117)
- Hao et al. 2024, *Coconut*, arXiv [2412.06769](https://arxiv.org/abs/2412.06769)
- Shai et al. 2024, *Transformers represent belief state geometry in their residual stream*, arXiv [2405.15943](https://arxiv.org/abs/2405.15943)
- Complementary learning systems and new-word consolidation, as cited by Sol: [PubMed 22141588](https://pubmed.ncbi.nlm.nih.gov/22141588/), [PMC2846311](https://pmc.ncbi.nlm.nih.gov/articles/PMC2846311/)
- Classical: Kalman (minimal realization); Moore 1981 (balanced truncation); Vygotsky, *Thought and Language*; Liberman (motor theory of speech perception, contested)
