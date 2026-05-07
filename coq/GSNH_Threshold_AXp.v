(*
  GSNH_Threshold_AXp.v

  Verified mathematical core for the GSNH-MDT threshold encoding.

  This file is intentionally small and conservative.

  What it proves:
  1. Numeric threshold literals are encoded as signed Boolean threshold atoms.
  2. GE(f,t) is encoded as B(f,t).
  3. LT(f,t) is encoded as not B(f,t).
  4. Predicate true branches are encoded as one disjunctive clause.
  5. Predicate false branches are encoded as singleton negated-literal clauses.
  6. Root-to-leaf path encoding is equivalent to direct numeric path semantics,
     under the Boolean assignment induced by the numeric valuation.
  7. Structural threshold order clauses are sound:
        if t1 <= t2 then B(f,t2) -> B(f,t1).
  8. Structural order clauses are both Horn and Anti-Horn.
  9. Horn and Anti-Horn syntactic restrictions are preserved for true-branch
     clauses.
  10. Weak-AXp opposite-path inconsistency is preserved by the encoding.

  Scope:
  - Thresholds are exact rationals Q.
  - This verifies the mathematical encoding, not the Python implementation.
  - This file does not prove benchmark accuracy or runtime measurements.
*)

Require Import Coq.QArith.QArith.
Require Import Coq.Lists.List.
Require Import Coq.Bool.Bool.
Require Import Coq.Arith.PeanoNat.

Import ListNotations.
Open Scope Q_scope.

(* ================================================================ *)
(* 1. Basic mathematical objects                                     *)
(* ================================================================ *)

Definition Feature := nat.
Definition Threshold := Q.

Definition Atom := (Feature * Threshold)%type.
Definition Valuation := Feature -> Q.

Inductive Polarity : Type :=
| GE
| LT.

Record Literal : Type := mkLiteral {
  lit_feature : Feature;
  lit_threshold : Threshold;
  lit_polarity : Polarity
}.

Definition atom_evalb (x : Valuation) (a : Atom) : bool :=
  let '(f,t) := a in Qle_bool t (x f).

Definition lit_evalb (x : Valuation) (l : Literal) : bool :=
  match l.(lit_polarity) with
  | GE => Qle_bool l.(lit_threshold) (x l.(lit_feature))
  | LT => negb (Qle_bool l.(lit_threshold) (x l.(lit_feature)))
  end.

(* ================================================================ *)
(* 2. Signed Boolean threshold atoms                                 *)
(* ================================================================ *)

Record SignedAtom : Type := mkSignedAtom {
  sa_atom : Atom;
  sa_positive : bool
}.

Definition Assignment := Atom -> bool.

Definition signed_evalb (rho : Assignment) (s : SignedAtom) : bool :=
  if s.(sa_positive)
  then rho s.(sa_atom)
  else negb (rho s.(sa_atom)).

Definition induced_assignment (x : Valuation) : Assignment :=
  fun a => atom_evalb x a.

Definition encode_literal (l : Literal) : SignedAtom :=
  mkSignedAtom
    (l.(lit_feature), l.(lit_threshold))
    (match l.(lit_polarity) with
     | GE => true
     | LT => false
     end).

Definition negate_signed (s : SignedAtom) : SignedAtom :=
  mkSignedAtom s.(sa_atom) (negb s.(sa_positive)).

Lemma encode_literal_correct :
  forall (x : Valuation) (l : Literal),
    signed_evalb (induced_assignment x) (encode_literal l)
    = lit_evalb x l.
Proof.
  intros x l.
  destruct l as [f t p].
  destruct p; reflexivity.
Qed.

Lemma negate_signed_correct :
  forall (rho : Assignment) (s : SignedAtom),
    signed_evalb rho (negate_signed s)
    = negb (signed_evalb rho s).
Proof.
  intros rho s.
  destruct s as [a pos].
  destruct pos.
  - reflexivity.
  - unfold negate_signed, signed_evalb; simpl.
    destruct (rho a); reflexivity.
Qed.

(* ================================================================ *)
(* 3. Clauses, CNF, and predicate branches                           *)
(* ================================================================ *)

Definition Clause := list SignedAtom.
Definition CNF := list Clause.
Definition Predicate := list Literal.

Definition clause_evalb (rho : Assignment) (c : Clause) : bool :=
  existsb (fun s => signed_evalb rho s) c.

Definition cnf_evalb (rho : Assignment) (cs : CNF) : bool :=
  forallb (fun c => clause_evalb rho c) cs.

Definition pred_evalb (x : Valuation) (p : Predicate) : bool :=
  existsb (fun l => lit_evalb x l) p.

Definition pred_false_evalb (x : Valuation) (p : Predicate) : bool :=
  forallb (fun l => negb (lit_evalb x l)) p.

Definition pred_true_branch_clause (p : Predicate) : Clause :=
  map encode_literal p.

Definition pred_false_branch_clauses (p : Predicate) : CNF :=
  map (fun l => [negate_signed (encode_literal l)]) p.

Lemma clause_singleton_eval :
  forall (rho : Assignment) (s : SignedAtom),
    clause_evalb rho [s] = signed_evalb rho s.
Proof.
  intros rho s.
  unfold clause_evalb.
  simpl.
  destruct (signed_evalb rho s); reflexivity.
Qed.

Lemma pred_true_branch_correct :
  forall (x : Valuation) (p : Predicate),
    clause_evalb (induced_assignment x) (pred_true_branch_clause p)
    = pred_evalb x p.
Proof.
  intros x p.
  unfold clause_evalb, pred_true_branch_clause, pred_evalb.
  induction p as [| l tl IH].
  - reflexivity.
  - simpl.
    rewrite encode_literal_correct.
    rewrite IH.
    reflexivity.
Qed.

Lemma pred_false_branch_correct :
  forall (x : Valuation) (p : Predicate),
    cnf_evalb (induced_assignment x) (pred_false_branch_clauses p)
    = pred_false_evalb x p.
Proof.
  intros x p.
  induction p as [| l tl IH].
  - reflexivity.
  - (* Unfold the top-level definitions carefully *)
    unfold pred_false_branch_clauses, pred_false_evalb.
    simpl.
    
    (* Unfold cnf_evalb at the head of the list *)
    unfold cnf_evalb at 1.
    simpl.
    
    (* At this point, Coq has aggressively simplified the clause evaluation 
       to: (signed_evalb ... || false) && (forallb ...)
       We can eliminate the '|| false' using the standard boolean lemma orb_false_r *)
    rewrite orb_false_r.
    
    (* Now apply our proven lemmas to the remaining signed_evalb term *)
    rewrite negate_signed_correct.
    rewrite encode_literal_correct.
    
    (* Fold the tail back into our definitions so it matches the IH *)
    change (forallb (fun c : Clause => clause_evalb (induced_assignment x) c)
              (map (fun l0 : Literal => [negate_signed (encode_literal l0)]) tl))
      with (cnf_evalb (induced_assignment x) (pred_false_branch_clauses tl)).
      
    (* Use the induction hypothesis *)
    rewrite IH.
    reflexivity.
Qed.

(* ================================================================ *)
(* 4. Path semantics and path encoding                               *)
(* ================================================================ *)

Definition Edge := (Predicate * bool)%type.
Definition Path := list Edge.

Definition edge_evalb (x : Valuation) (e : Edge) : bool :=
  let '(p, branch) := e in
  if branch
  then pred_evalb x p
  else pred_false_evalb x p.

Definition edge_clauses (e : Edge) : CNF :=
  let '(p, branch) := e in
  if branch
  then [pred_true_branch_clause p]
  else pred_false_branch_clauses p.

Fixpoint path_evalb (x : Valuation) (path : Path) : bool :=
  match path with
  | [] => true
  | e :: tl => andb (edge_evalb x e) (path_evalb x tl)
  end.

Fixpoint path_clauses (path : Path) : CNF :=
  match path with
  | [] => []
  | e :: tl => edge_clauses e ++ path_clauses tl
  end.

Lemma cnf_evalb_app :
  forall (rho : Assignment) (a b : CNF),
    cnf_evalb rho (a ++ b)
    = andb (cnf_evalb rho a) (cnf_evalb rho b).
Proof.
  intros rho a b.
  unfold cnf_evalb.
  apply forallb_app.
Qed.

Lemma edge_encoding_correct :
  forall (x : Valuation) (e : Edge),
    cnf_evalb (induced_assignment x) (edge_clauses e)
    = edge_evalb x e.
Proof.
  intros x e.
  destruct e as [p branch].
  destruct branch.
  - simpl. unfold cnf_evalb. simpl.
    rewrite andb_true_r.
    apply pred_true_branch_correct.
  - simpl. apply pred_false_branch_correct.
Qed.

Theorem path_encoding_correct :
  forall (x : Valuation) (path : Path),
    cnf_evalb (induced_assignment x) (path_clauses path)
    = path_evalb x path.
Proof.
  intros x path.
  induction path as [| e tl IH].
  - reflexivity.
  - simpl.
    rewrite cnf_evalb_app.
    rewrite edge_encoding_correct.
    rewrite IH.
    reflexivity.
Qed.

(* ================================================================ *)
(* 5. Structural threshold order clauses                             *)
(* ================================================================ *)

Definition structural_order_clause
           (f : Feature) (t1 t2 : Threshold) : Clause :=
  [
    mkSignedAtom (f,t2) false;
    mkSignedAtom (f,t1) true
  ].

Theorem structural_order_clause_sound :
  forall (x : Valuation) (f : Feature) (t1 t2 : Threshold),
    (t1 <= t2)%Q ->
    clause_evalb (induced_assignment x)
                 (structural_order_clause f t1 t2) = true.
Proof.
  intros x f t1 t2 Hle.
  unfold structural_order_clause.
  unfold clause_evalb.
  simpl.
  unfold signed_evalb.
  unfold induced_assignment.
  unfold atom_evalb.
  simpl.
  destruct (Qle_bool t2 (x f)) eqn:E2.
  - simpl.
    assert (Qle_bool t1 (x f) = true) as E1.
    {
      apply Qle_bool_iff.
      apply Qle_bool_iff in E2.
      eapply Qle_trans.
      + exact Hle.
      + exact E2.
    }
    rewrite E1.
    reflexivity.
  - reflexivity.
Qed.

(* ================================================================ *)
(* 6. Horn and Anti-Horn syntactic shape                             *)
(* ================================================================ *)

Fixpoint count_signed_pos (c : Clause) : nat :=
  match c with
  | [] => 0%nat
  | s :: tl =>
      ((if s.(sa_positive) then 1%nat else 0%nat)
       + count_signed_pos tl)%nat
  end.

Fixpoint count_signed_neg (c : Clause) : nat :=
  match c with
  | [] => 0%nat
  | s :: tl =>
      ((if s.(sa_positive) then 0%nat else 1%nat)
       + count_signed_neg tl)%nat
  end.

Definition is_horn_clause (c : Clause) : bool :=
  Nat.leb (count_signed_pos c) 1.

Definition is_antihorn_clause (c : Clause) : bool :=
  Nat.leb (count_signed_neg c) 1.

Theorem structural_order_clause_is_horn :
  forall (f : Feature) (t1 t2 : Threshold),
    is_horn_clause (structural_order_clause f t1 t2) = true.
Proof.
  intros f t1 t2.
  unfold is_horn_clause.
  unfold structural_order_clause.
  simpl.
  reflexivity.
Qed.

Theorem structural_order_clause_is_antihorn :
  forall (f : Feature) (t1 t2 : Threshold),
    is_antihorn_clause (structural_order_clause f t1 t2) = true.
Proof.
  intros f t1 t2.
  unfold is_antihorn_clause.
  unfold structural_order_clause.
  simpl.
  reflexivity.
Qed.

Fixpoint count_ge_literals (p : Predicate) : nat :=
  match p with
  | [] => 0%nat
  | l :: tl =>
      ((match l.(lit_polarity) with
        | GE => 1%nat
        | LT => 0%nat
        end)
       + count_ge_literals tl)%nat
  end.

Fixpoint count_lt_literals (p : Predicate) : nat :=
  match p with
  | [] => 0%nat
  | l :: tl =>
      ((match l.(lit_polarity) with
        | GE => 0%nat
        | LT => 1%nat
        end)
       + count_lt_literals tl)%nat
  end.

Definition is_horn_predicate (p : Predicate) : bool :=
  Nat.leb (count_ge_literals p) 1.

Definition is_antihorn_predicate (p : Predicate) : bool :=
  Nat.leb (count_lt_literals p) 1.

Lemma count_pos_encoded :
  forall (p : Predicate),
    count_signed_pos (pred_true_branch_clause p)
    = count_ge_literals p.
Proof.
  intros p.
  unfold pred_true_branch_clause.
  induction p as [| l tl IH].
  - reflexivity.
  - destruct l as [f t pol].
    destruct pol; simpl; rewrite IH; reflexivity.
Qed.

Lemma count_neg_encoded :
  forall (p : Predicate),
    count_signed_neg (pred_true_branch_clause p)
    = count_lt_literals p.
Proof.
  intros p.
  unfold pred_true_branch_clause.
  induction p as [| l tl IH].
  - reflexivity.
  - destruct l as [f t pol].
    destruct pol; simpl; rewrite IH; reflexivity.
Qed.

Theorem horn_predicate_true_branch_is_horn :
  forall (p : Predicate),
    is_horn_predicate p = true ->
    is_horn_clause (pred_true_branch_clause p) = true.
Proof.
  intros p H.
  unfold is_horn_predicate in H.
  unfold is_horn_clause.
  rewrite count_pos_encoded.
  exact H.
Qed.

Theorem antihorn_predicate_true_branch_is_antihorn :
  forall (p : Predicate),
    is_antihorn_predicate p = true ->
    is_antihorn_clause (pred_true_branch_clause p) = true.
Proof.
  intros p H.
  unfold is_antihorn_predicate in H.
  unfold is_antihorn_clause.
  rewrite count_neg_encoded.
  exact H.
Qed.

Lemma singleton_clause_is_horn :
  forall (s : SignedAtom),
    is_horn_clause [s] = true.
Proof.
  intros s.
  destruct s as [a pos].
  destruct pos; reflexivity.
Qed.

Lemma singleton_clause_is_antihorn :
  forall (s : SignedAtom),
    is_antihorn_clause [s] = true.
Proof.
  intros s.
  destruct s as [a pos].
  destruct pos; reflexivity.
Qed.

Definition all_horn_clauses (cs : CNF) : bool :=
  forallb is_horn_clause cs.

Definition all_antihorn_clauses (cs : CNF) : bool :=
  forallb is_antihorn_clause cs.

Theorem false_branch_clauses_are_horn :
  forall (p : Predicate),
    all_horn_clauses (pred_false_branch_clauses p) = true.
Proof.
  intros p.
  unfold all_horn_clauses, pred_false_branch_clauses.
  induction p as [| l tl IH].
  - reflexivity.
  - simpl.
    rewrite singleton_clause_is_horn.
    simpl. exact IH.
Qed.

Theorem false_branch_clauses_are_antihorn :
  forall (p : Predicate),
    all_antihorn_clauses (pred_false_branch_clauses p) = true.
Proof.
  intros p.
  unfold all_antihorn_clauses, pred_false_branch_clauses.
  induction p as [| l tl IH].
  - reflexivity.
  - simpl.
    rewrite singleton_clause_is_antihorn.
    simpl. exact IH.
Qed.

(* ================================================================ *)
(* 7. Weak-AXp opposite-path inconsistency                           *)
(* ================================================================ *)

Definition weak_axp_direct_evalb
           (x : Valuation) (opposite_paths : list Path) : bool :=
  forallb (fun p => negb (path_evalb x p)) opposite_paths.

Definition weak_axp_encoded_evalb
           (x : Valuation) (opposite_paths : list Path) : bool :=
  forallb
    (fun p =>
       negb (cnf_evalb (induced_assignment x) (path_clauses p)))
    opposite_paths.

Theorem weak_axp_encoding_correct :
  forall (x : Valuation) (opposite_paths : list Path),
    weak_axp_encoded_evalb x opposite_paths
    = weak_axp_direct_evalb x opposite_paths.
Proof.
  intros x opposite_paths.
  unfold weak_axp_encoded_evalb, weak_axp_direct_evalb.
  induction opposite_paths as [| p tl IH].
  - reflexivity.
  - simpl.
    rewrite path_encoding_correct.
    rewrite IH.
    reflexivity.
Qed.

(* ================================================================ *)
(* 8. Small concrete sanity examples                                 *)
(* ================================================================ *)

Definition sample_x : Valuation :=
  fun f =>
    match f with
    | O => (7#1)
    | S O => (1#1)
    | _ => (0#1)
    end.

Definition lit_x0_ge_5 : Literal :=
  mkLiteral 0%nat (5#1) GE.

Definition lit_x1_lt_2 : Literal :=
  mkLiteral 1%nat (2#1) LT.

Definition sample_pred : Predicate :=
  [lit_x0_ge_5; lit_x1_lt_2].

Example sample_literal_ge_true :
  lit_evalb sample_x lit_x0_ge_5 = true.
Proof.
  reflexivity.
Qed.

Example sample_literal_lt_true :
  lit_evalb sample_x lit_x1_lt_2 = true.
Proof.
  reflexivity.
Qed.

Example sample_pred_true :
  pred_evalb sample_x sample_pred = true.
Proof.
  reflexivity.
Qed.

Example sample_true_branch_encoding :
  clause_evalb (induced_assignment sample_x)
               (pred_true_branch_clause sample_pred) = true.
Proof.
  reflexivity.
Qed.

Example sample_path_encoding_direct :
  path_evalb sample_x [(sample_pred, true)] = true.
Proof.
  reflexivity.
Qed.

Example sample_path_encoding_encoded :
  cnf_evalb (induced_assignment sample_x)
            (path_clauses [(sample_pred, true)]) = true.
Proof.
  reflexivity.
Qed.