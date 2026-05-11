(*
  GSNH_Threshold_Completeness.v

  Formal verification layer for the GSNH-MDT structural threshold encoding.

  This file contains two layers.

  Layer 1:
    Local encoding correctness.
    For any concrete numeric valuation x, direct numeric path evaluation is
    equal to Boolean CNF evaluation under the assignment induced by x.

  Layer 2:
    Threshold-order completeness.
    For a finite set of threshold atoms, every Boolean assignment satisfying
    the threshold monotonicity/order condition can be realized by a numeric
    valuation over Q.

  Main new theorem:
    order_consistent_assignment_has_numeric_realizer_for_path

  Meaning:
    For every path and every Boolean assignment rho over its threshold atoms,
    if rho respects threshold order, then there exists a numeric valuation x
    that realizes rho on all atoms appearing in that path.

  Scope:
    - Exact rational thresholds Q.
    - Horn/Anti-Horn threshold-atom mathematics.
    - Not a verification of Python code or benchmark accuracy.
*)

Require Import Coq.QArith.QArith.
Require Import Coq.Lists.List.
Require Import Coq.Bool.Bool.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.micromega.Lqa.

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
  - unfold pred_false_branch_clauses, pred_false_evalb.
    simpl.
    unfold cnf_evalb at 1.
    simpl.
    rewrite orb_false_r.
    rewrite negate_signed_correct.
    rewrite encode_literal_correct.
    change (forallb (fun c : Clause => clause_evalb (induced_assignment x) c)
              (map (fun l0 : Literal => [negate_signed (encode_literal l0)]) tl))
      with (cnf_evalb (induced_assignment x) (pred_false_branch_clauses tl)).
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
(* 8. Threshold-order completeness                                   *)
(* ================================================================ *)

(* This section proves the next missing mathematical layer.

   If a finite Boolean assignment over threshold atoms respects the natural
   monotonicity of threshold atoms, then it can be realized by a concrete
   numeric valuation.

   Monotonicity/order condition:
       if t1 <= t2 and B(f,t2) is true, then B(f,t1) must be true.

   This exactly captures the structural threshold order clauses:
       B(f,t2) -> B(f,t1)
*)

(* Some Coq installations do not expose Qle_dec directly.
   We therefore define max/min using Qle_bool, which is available
   from Coq.QArith.QArith. *)

Definition qmax2 (a b : Q) : Q :=
  if Qle_bool a b then b else a.

Definition qmin2 (a b : Q) : Q :=
  if Qle_bool a b then a else b.

Lemma Qle_bool_false_not_le :
  forall a b : Q,
    Qle_bool a b = false -> ~ (a <= b).
Proof.
  intros a b Hfalse Hle.
  apply Qle_bool_iff in Hle.
  rewrite Hle in Hfalse.
  discriminate.
Qed.

Lemma qmax2_ge_l :
  forall a b : Q, a <= qmax2 a b.
Proof.
  intros a b.
  unfold qmax2.
  destruct (Qle_bool a b) eqn:E.
  - apply Qle_bool_iff. exact E.
  - lra.
Qed.

Lemma qmax2_ge_r :
  forall a b : Q, b <= qmax2 a b.
Proof.
  intros a b.
  unfold qmax2.
  destruct (Qle_bool a b) eqn:E.
  - lra.
  - assert (~ a <= b) as Hn.
    { apply Qle_bool_false_not_le. exact E. }
    lra.
Qed.

Lemma qmin2_le_l :
  forall a b : Q, qmin2 a b <= a.
Proof.
  intros a b.
  unfold qmin2.
  destruct (Qle_bool a b) eqn:E.
  - lra.
  - assert (~ a <= b) as Hn.
    { apply Qle_bool_false_not_le. exact E. }
    lra.
Qed.

Lemma qmin2_le_r :
  forall a b : Q, qmin2 a b <= b.
Proof.
  intros a b.
  unfold qmin2.
  destruct (Qle_bool a b) eqn:E.
  - apply Qle_bool_iff. exact E.
  - lra.
Qed.

Lemma qmax2_choice :
  forall a b : Q, qmax2 a b = a \/ qmax2 a b = b.
Proof.
  intros a b.
  unfold qmax2.
  destruct (Qle_bool a b) eqn:E.
  - right. reflexivity.
  - left. reflexivity.
Qed.

Fixpoint qmax_list_with (d : Q) (l : list Q) : Q :=
  match l with
  | [] => d
  | x :: xs => qmax2 x (qmax_list_with d xs)
  end.

Definition qmax_list (l : list Q) : option Q :=
  match l with
  | [] => None
  | x :: xs => Some (qmax_list_with x xs)
  end.

Lemma qmax_list_none_no_in :
  forall (l : list Q) (t : Q),
    qmax_list l = None ->
    ~ In t l.
Proof.
  intros l t Hnone Hin.
  destruct l as [| x xs].
  - simpl in Hin. exact Hin.
  - simpl in Hnone. discriminate.
Qed.

Lemma qmax_list_with_ge_default :
  forall (d : Q) (l : list Q),
    d <= qmax_list_with d l.
Proof.
  intros d l.
  induction l as [| x xs IH].
  - simpl. lra.
  - simpl.
    eapply Qle_trans.
    + exact IH.
    + apply qmax2_ge_r.
Qed.

Lemma qmax_list_with_ge_in :
  forall (d : Q) (l : list Q) (t : Q),
    In t l ->
    t <= qmax_list_with d l.
Proof.
  intros d l t Hin.
  induction l as [| x xs IH].
  - contradiction.
  - simpl in *.
    destruct Hin as [Hx | Hin].
    + subst x. apply qmax2_ge_l.
    + eapply Qle_trans.
      * apply IH. exact Hin.
      * apply qmax2_ge_r.
Qed.

Lemma qmax_list_ge :
  forall (l : list Q) (m t : Q),
    qmax_list l = Some m ->
    In t l ->
    t <= m.
Proof.
  intros l m t Hmax Hin.
  destruct l as [| x xs].
  - inversion Hin.
  - simpl in Hmax.
    inversion Hmax; subst; clear Hmax.
    simpl in Hin.
    destruct Hin as [Hx | Hin].
    + subst x. apply qmax_list_with_ge_default.
    + apply qmax_list_with_ge_in. exact Hin.
Qed.

Lemma qmax_list_with_in :
  forall (d : Q) (l : list Q),
    In (qmax_list_with d l) (d :: l).
Proof.
  intros d l.
  induction l as [| x xs IH].
  - simpl. left. reflexivity.
  - simpl.
    destruct (qmax2_choice x (qmax_list_with d xs)) as [H | H].
    + rewrite H. right. left. reflexivity.
    + rewrite H.
      simpl in IH.
      destruct IH as [IH | IH].
      * left. exact IH.
      * right. right. exact IH.
Qed.

Lemma qmax_list_in :
  forall (l : list Q) (m : Q),
    qmax_list l = Some m ->
    In m l.
Proof.
  intros l m H.
  destruct l as [| x xs].
  - simpl in H. discriminate.
  - simpl in H.
    inversion H; subst; clear H.
    apply qmax_list_with_in.
Qed.

Fixpoint qmin_list_with (d : Q) (l : list Q) : Q :=
  match l with
  | [] => d
  | x :: xs => qmin2 x (qmin_list_with d xs)
  end.

Definition qmin_list (l : list Q) : option Q :=
  match l with
  | [] => None
  | x :: xs => Some (qmin_list_with x xs)
  end.

Lemma qmin_list_none_no_in :
  forall (l : list Q) (t : Q),
    qmin_list l = None ->
    ~ In t l.
Proof.
  intros l t Hnone Hin.
  destruct l as [| x xs].
  - simpl in Hin. exact Hin.
  - simpl in Hnone. discriminate.
Qed.

Lemma qmin_list_with_le_default :
  forall (d : Q) (l : list Q),
    qmin_list_with d l <= d.
Proof.
  intros d l.
  induction l as [| x xs IH].
  - simpl. lra.
  - simpl.
    eapply Qle_trans.
    + apply qmin2_le_r.
    + exact IH.
Qed.

Lemma qmin_list_with_le_in :
  forall (d : Q) (l : list Q) (t : Q),
    In t l ->
    qmin_list_with d l <= t.
Proof.
  intros d l t Hin.
  induction l as [| x xs IH].
  - contradiction.
  - simpl in *.
    destruct Hin as [Hx | Hin].
    + subst x. apply qmin2_le_l.
    + eapply Qle_trans.
      * apply qmin2_le_r.
      * apply IH. exact Hin.
Qed.

Lemma qmin_list_le :
  forall (l : list Q) (m t : Q),
    qmin_list l = Some m ->
    In t l ->
    m <= t.
Proof.
  intros l m t Hmin Hin.
  destruct l as [| x xs].
  - inversion Hin.
  - simpl in Hmin.
    inversion Hmin; subst; clear Hmin.
    simpl in Hin.
    destruct Hin as [Hx | Hin].
    + subst x. apply qmin_list_with_le_default.
    + apply qmin_list_with_le_in. exact Hin.
Qed.

(* Extract the thresholds of one feature from a finite atom set. *)
Fixpoint thresholds_for (atoms : list Atom) (f : Feature) : list Threshold :=
  match atoms with
  | [] => []
  | (g,t) :: tl =>
      if Nat.eqb g f
      then t :: thresholds_for tl f
      else thresholds_for tl f
  end.

Lemma thresholds_for_in_intro :
  forall (atoms : list Atom) (f : Feature) (t : Threshold),
    In (f,t) atoms ->
    In t (thresholds_for atoms f).
Proof.
  intros atoms f t Hin.
  induction atoms as [| [g u] tl IH].
  - contradiction.
  - simpl in *.
    destruct Hin as [Hhead | Htail].
    + inversion Hhead; subst.
      rewrite Nat.eqb_refl.
      simpl. left. reflexivity.
    + destruct (Nat.eqb g f) eqn:E.
      * simpl. right. apply IH. exact Htail.
      * apply IH. exact Htail.
Qed.

Lemma thresholds_for_in_elim :
  forall (atoms : list Atom) (f : Feature) (t : Threshold),
    In t (thresholds_for atoms f) ->
    In (f,t) atoms.
Proof.
  intros atoms f t Hin.
  induction atoms as [| [g u] tl IH].
  - contradiction.
  - simpl in *.
    destruct (Nat.eqb g f) eqn:E.
    + simpl in Hin.
      destruct Hin as [Hhead | Htail].
      * subst u.
        apply Nat.eqb_eq in E.
        subst g.
        left. reflexivity.
      * right. apply IH. exact Htail.
    + right. apply IH. exact Hin.
Qed.

Definition true_thresholds
           (rho : Assignment) (atoms : list Atom) (f : Feature)
           : list Threshold :=
  filter (fun t => rho (f,t)) (thresholds_for atoms f).

Lemma true_thresholds_intro :
  forall (rho : Assignment) (atoms : list Atom)
         (f : Feature) (t : Threshold),
    In (f,t) atoms ->
    rho (f,t) = true ->
    In t (true_thresholds rho atoms f).
Proof.
  intros rho atoms f t Hin Htrue.
  unfold true_thresholds.
  apply filter_In.
  split.
  - apply thresholds_for_in_intro. exact Hin.
  - exact Htrue.
Qed.

Lemma true_thresholds_in_atom :
  forall (rho : Assignment) (atoms : list Atom)
         (f : Feature) (t : Threshold),
    In t (true_thresholds rho atoms f) ->
    In (f,t) atoms.
Proof.
  intros rho atoms f t Hin.
  unfold true_thresholds in Hin.
  apply filter_In in Hin.
  destruct Hin as [Hin _].
  apply thresholds_for_in_elim. exact Hin.
Qed.

Lemma true_thresholds_in_true :
  forall (rho : Assignment) (atoms : list Atom)
         (f : Feature) (t : Threshold),
    In t (true_thresholds rho atoms f) ->
    rho (f,t) = true.
Proof.
  intros rho atoms f t Hin.
  unfold true_thresholds in Hin.
  apply filter_In in Hin.
  destruct Hin as [_ Htrue].
  exact Htrue.
Qed.

(* Order consistency over a finite atom set. *)
Definition order_consistent_on
           (atoms : list Atom) (rho : Assignment) : Prop :=
  forall (f : Feature) (t1 t2 : Threshold),
    In (f,t1) atoms ->
    In (f,t2) atoms ->
    t1 <= t2 ->
    rho (f,t2) = true ->
    rho (f,t1) = true.

(* The canonical valuation induced by an order-consistent Boolean assignment.

   For each feature:
   - if there are true thresholds, choose the maximum true threshold;
   - if there are no true thresholds but some thresholds exist, choose
     one less than the minimum threshold;
   - if the feature is absent, choose 0.
*)
Definition canonical_value
           (atoms : list Atom) (rho : Assignment) (f : Feature) : Q :=
  match qmax_list (true_thresholds rho atoms f) with
  | Some m => m
  | None =>
      match qmin_list (thresholds_for atoms f) with
      | Some m => m - 1
      | None => 0
      end
  end.

Definition canonical_valuation
           (atoms : list Atom) (rho : Assignment) : Valuation :=
  fun f => canonical_value atoms rho f.

Theorem canonical_valuation_realizes_assignment :
  forall (atoms : list Atom) (rho : Assignment),
    order_consistent_on atoms rho ->
    forall (f : Feature) (t : Threshold),
      In (f,t) atoms ->
      atom_evalb (canonical_valuation atoms rho) (f,t)
      = rho (f,t).
Proof.
  intros atoms rho Hord f t HinAtom.
  unfold canonical_valuation, canonical_value, atom_evalb.
  destruct (rho (f,t)) eqn:Hrho.

  - (* Case 1: rho says B(f,t) is true. *)
    assert (HinTrue : In t (true_thresholds rho atoms f)).
    {
      apply true_thresholds_intro.
      - exact HinAtom.
      - exact Hrho.
    }
    destruct (qmax_list (true_thresholds rho atoms f)) as [m |] eqn:Hmax.
    + apply Qle_bool_iff.
      eapply qmax_list_ge.
      * exact Hmax.
      * exact HinTrue.
    + exfalso.
      eapply qmax_list_none_no_in.
      * exact Hmax.
      * exact HinTrue.

  - (* Case 2: rho says B(f,t) is false. *)
    destruct (qmax_list (true_thresholds rho atoms f)) as [m |] eqn:Hmax.

    + (* There exists a true threshold m. If t <= m, monotonicity would force
         B(f,t) true, contradiction. *)
      assert (HmInTrue : In m (true_thresholds rho atoms f)).
      {
        eapply qmax_list_in.
        exact Hmax.
      }
      assert (HmAtom : In (f,m) atoms).
      {
        apply true_thresholds_in_atom with (rho := rho).
        exact HmInTrue.
      }
      assert (HmTrue : rho (f,m) = true).
      {
        apply true_thresholds_in_true with (atoms := atoms).
        exact HmInTrue.
      }
      destruct (Qle_bool t m) eqn:E.
      * apply Qle_bool_iff in E.
        specialize (Hord f t m HinAtom HmAtom E HmTrue).
        rewrite Hrho in Hord.
        discriminate.
      * reflexivity.

    + (* There are no true thresholds for f. Choose below the minimum threshold. *)
      assert (HinAll : In t (thresholds_for atoms f)).
      {
        apply thresholds_for_in_intro.
        exact HinAtom.
      }
      destruct (qmin_list (thresholds_for atoms f)) as [mi |] eqn:Hmin.
      * destruct (Qle_bool t (mi - 1)) eqn:E.
        -- apply Qle_bool_iff in E.
           assert (Hmi_le_t : mi <= t).
           {
             eapply qmin_list_le.
             + exact Hmin.
             + exact HinAll.
           }
           exfalso. lra.
        -- reflexivity.
      * exfalso.
        eapply qmin_list_none_no_in.
        -- exact Hmin.
        -- exact HinAll.
Qed.

(* Atoms appearing in predicates and paths. *)

Definition atom_of_literal (l : Literal) : Atom :=
  (l.(lit_feature), l.(lit_threshold)).

Fixpoint atoms_of_predicate (p : Predicate) : list Atom :=
  match p with
  | [] => []
  | l :: tl => atom_of_literal l :: atoms_of_predicate tl
  end.

Fixpoint atoms_of_path (path : Path) : list Atom :=
  match path with
  | [] => []
  | (p, _) :: tl => atoms_of_predicate p ++ atoms_of_path tl
  end.

Theorem order_consistent_assignment_has_numeric_realizer_for_path :
  forall (path : Path) (rho : Assignment),
    order_consistent_on (atoms_of_path path) rho ->
    exists x : Valuation,
      forall (a : Atom),
        In a (atoms_of_path path) ->
        atom_evalb x a = rho a.
Proof.
  intros path rho Hord.
  exists (canonical_valuation (atoms_of_path path) rho).
  intros [f t] Hin.
  apply canonical_valuation_realizes_assignment.
  - exact Hord.
  - exact Hin.
Qed.

(* The numeric assignment induced by any concrete valuation is always
   order-consistent. This proves the easy direction:
   numeric model -> order-consistent Boolean model.
*)
Theorem induced_assignment_order_consistent :
  forall (atoms : list Atom) (x : Valuation),
    order_consistent_on atoms (induced_assignment x).
Proof.
  intros atoms x.
  unfold order_consistent_on.
  intros f t1 t2 Hin1 Hin2 Hle Htrue2.
  unfold induced_assignment, atom_evalb in *.
  apply Qle_bool_iff.
  apply Qle_bool_iff in Htrue2.
  eapply Qle_trans.
  - exact Hle.
  - exact Htrue2.
Qed.

Definition numeric_path_satisfiable (path : Path) : Prop :=
  exists x : Valuation, path_evalb x path = true.

Definition boolean_path_satisfiable_with_order (path : Path) : Prop :=
  exists rho : Assignment,
    order_consistent_on (atoms_of_path path) rho /\
    cnf_evalb rho (path_clauses path) = true.

Theorem numeric_path_sat_implies_boolean_path_sat :
  forall path : Path,
    numeric_path_satisfiable path ->
    boolean_path_satisfiable_with_order path.
Proof.
  intros path [x Hsat].
  exists (induced_assignment x).
  split.
  - apply induced_assignment_order_consistent.
  - rewrite path_encoding_correct.
    exact Hsat.
Qed.

(* ================================================================ *)
(* 9. Boolean path satisfiability implies numeric path satisfiability *)
(* ================================================================ *)

(* This section proves the reverse direction.

   If the Boolean threshold encoding of a path is satisfiable by an
   order-consistent assignment rho, then there exists a numeric valuation x
   satisfying the original numeric path.

   The proof uses the canonical valuation constructed above.
*)

Lemma signed_evalb_agree :
  forall (rho1 rho2 : Assignment) (s : SignedAtom),
    rho1 s.(sa_atom) = rho2 s.(sa_atom) ->
    signed_evalb rho1 s = signed_evalb rho2 s.
Proof.
  intros rho1 rho2 [a pos] Hagree.
  simpl in *.
  destruct pos.
  - exact Hagree.
  - apply f_equal with (f := negb) in Hagree.
    exact Hagree.
Qed.

Lemma clause_evalb_agree :
  forall (rho1 rho2 : Assignment) (c : Clause),
    (forall s : SignedAtom,
        In s c ->
        rho1 s.(sa_atom) = rho2 s.(sa_atom)) ->
    clause_evalb rho1 c = clause_evalb rho2 c.
Proof.
  intros rho1 rho2 c Hagree.
  unfold clause_evalb.
  induction c as [| s tl IH].
  - reflexivity.
  - simpl.
    rewrite (signed_evalb_agree rho1 rho2 s).
    + rewrite IH.
      * reflexivity.
      * intros s' Hin.
        apply Hagree.
        right. exact Hin.
    + apply Hagree.
      left. reflexivity.
Qed.

Lemma cnf_evalb_agree :
  forall (rho1 rho2 : Assignment) (cs : CNF),
    (forall (c : Clause) (s : SignedAtom),
        In c cs ->
        In s c ->
        rho1 s.(sa_atom) = rho2 s.(sa_atom)) ->
    cnf_evalb rho1 cs = cnf_evalb rho2 cs.
Proof.
  intros rho1 rho2 cs Hagree.
  unfold cnf_evalb.
  induction cs as [| c tl IH].
  - reflexivity.
  - simpl.
    rewrite (clause_evalb_agree rho1 rho2 c).
    + rewrite IH.
      * reflexivity.
      * intros c' s HinC HinS.
        exact (Hagree c' s (or_intror HinC) HinS).
    + intros s HinS.
      exact (Hagree c s (or_introl eq_refl) HinS).
Qed.

Lemma in_true_branch_clause_atom :
  forall (p : Predicate) (s : SignedAtom),
    In s (pred_true_branch_clause p) ->
    In s.(sa_atom) (atoms_of_predicate p).
Proof.
  intros p s Hin.
  unfold pred_true_branch_clause in Hin.
  induction p as [| l tl IH].
  - simpl in Hin. contradiction.
  - simpl in Hin.
    destruct Hin as [Hhead | Htail].
    + subst s.
      simpl. left. reflexivity.
    + simpl. right.
      apply IH. exact Htail.
Qed.

Lemma in_false_branch_clauses_atom :
  forall (p : Predicate) (c : Clause) (s : SignedAtom),
    In c (pred_false_branch_clauses p) ->
    In s c ->
    In s.(sa_atom) (atoms_of_predicate p).
Proof.
  induction p as [| l tl IH]; intros c s Hc Hs.
  - simpl in Hc. contradiction.
  - simpl in Hc.
    destruct Hc as [Hhead | Htail].
    + subst c.
      simpl in Hs.
      destruct Hs as [Hs | Hs].
      * subst s.
        simpl. left. reflexivity.
      * contradiction.
    + simpl. right.
      exact (IH c s Htail Hs).
Qed.

Lemma in_edge_clauses_atom :
  forall (p : Predicate) (branch : bool) (c : Clause) (s : SignedAtom),
    In c (edge_clauses (p, branch)) ->
    In s c ->
    In s.(sa_atom) (atoms_of_predicate p).
Proof.
  intros p branch c s Hc Hs.
  destruct branch.
  - simpl in Hc.
    destruct Hc as [Hhead | Htail].
    + subst c.
      apply in_true_branch_clause_atom.
      exact Hs.
    + contradiction.
  - simpl in Hc.
    apply in_false_branch_clauses_atom with (c := c).
    + exact Hc.
    + exact Hs.
Qed.

Lemma in_path_clauses_atom :
  forall (path : Path) (c : Clause) (s : SignedAtom),
    In c (path_clauses path) ->
    In s c ->
    In s.(sa_atom) (atoms_of_path path).
Proof.
  intros path c s Hc Hs.
  induction path as [| [p branch] tl IH].
  - simpl in Hc. contradiction.
  - simpl in Hc.
    apply in_app_or in Hc.
    destruct Hc as [Hedge | Htail].
    + simpl.
      apply in_or_app.
      left.
      apply in_edge_clauses_atom with (branch := branch) (c := c).
      * exact Hedge.
      * exact Hs.
    + simpl.
      apply in_or_app.
      right.
      apply IH.
      exact Htail.
Qed.

Lemma path_clauses_assignment_agree :
  forall (path : Path) (rho1 rho2 : Assignment),
    (forall a : Atom,
        In a (atoms_of_path path) ->
        rho1 a = rho2 a) ->
    cnf_evalb rho1 (path_clauses path)
    = cnf_evalb rho2 (path_clauses path).
Proof.
  intros path rho1 rho2 Hagree.
  apply cnf_evalb_agree.
  intros c s Hc Hs.
  apply Hagree.
  eapply in_path_clauses_atom.
  - exact Hc.
  - exact Hs.
Qed.

Theorem boolean_path_sat_implies_numeric_path_sat :
  forall path : Path,
    boolean_path_satisfiable_with_order path ->
    numeric_path_satisfiable path.
Proof.
  intros path [rho [Hord Hsat]].
  destruct
    (order_consistent_assignment_has_numeric_realizer_for_path
       path rho Hord)
    as [x Hreal].
  exists x.
  rewrite <- path_encoding_correct.
  rewrite (path_clauses_assignment_agree
             path
             (induced_assignment x)
             rho).
  - exact Hsat.
  - intros [f t] Hin.
    apply Hreal.
    exact Hin.
Qed.

Theorem path_satisfiability_equivalence :
  forall path : Path,
    numeric_path_satisfiable path <->
    boolean_path_satisfiable_with_order path.
Proof.
  intros path.
  split.
  - apply numeric_path_sat_implies_boolean_path_sat.
  - apply boolean_path_sat_implies_numeric_path_sat.
Qed.

(* ================================================================ *)
(* 10. Opposite-path blocking and weak-AXp path condition             *)
(* ================================================================ *)

(* In the MDT explanation algorithm, a candidate explanation is weakly
   sufficient when every opposite-class root-to-leaf path is impossible
   under the current partial assignment.

   This section formalizes the path-level core of that idea.

   numeric_opposite_paths_blocked:
     every opposite path is numerically unsatisfiable.

   boolean_opposite_paths_blocked:
     every opposite path has no order-consistent Boolean model.

   By the path satisfiability equivalence proved above, these two notions
   are equivalent.
*)

Definition numeric_opposite_paths_blocked
           (opposite_paths : list Path) : Prop :=
  forall p : Path,
    In p opposite_paths ->
    ~ numeric_path_satisfiable p.

Definition boolean_opposite_paths_blocked
           (opposite_paths : list Path) : Prop :=
  forall p : Path,
    In p opposite_paths ->
    ~ boolean_path_satisfiable_with_order p.

Theorem opposite_path_blocking_equivalence :
  forall opposite_paths : list Path,
    numeric_opposite_paths_blocked opposite_paths <->
    boolean_opposite_paths_blocked opposite_paths.
Proof.
  intros opposite_paths.
  split.
  - intros Hnum p Hin Hboolsat.
    apply (Hnum p Hin).
    apply boolean_path_sat_implies_numeric_path_sat.
    exact Hboolsat.
  - intros Hbool p Hin Hnumsat.
    apply (Hbool p Hin).
    apply numeric_path_sat_implies_boolean_path_sat.
    exact Hnumsat.
Qed.

(* Naming the same theorem in AXp language.

   This is still a path-level weak-AXp theorem: it assumes the opposite
   paths already include the constraints imposed by the candidate partial
   assignment. A later file can model feature subsets S and add their fixed
   assignment clauses explicitly.
*)

Definition weak_axp_path_condition_numeric
           (opposite_paths : list Path) : Prop :=
  numeric_opposite_paths_blocked opposite_paths.

Definition weak_axp_path_condition_boolean
           (opposite_paths : list Path) : Prop :=
  boolean_opposite_paths_blocked opposite_paths.

Theorem weak_axp_path_condition_equivalence :
  forall opposite_paths : list Path,
    weak_axp_path_condition_numeric opposite_paths <->
    weak_axp_path_condition_boolean opposite_paths.
Proof.
  intros opposite_paths.
  unfold weak_axp_path_condition_numeric.
  unfold weak_axp_path_condition_boolean.
  apply opposite_path_blocking_equivalence.
Qed.

Theorem boolean_blocking_implies_numeric_weak_axp_path_condition :
  forall opposite_paths : list Path,
    boolean_opposite_paths_blocked opposite_paths ->
    weak_axp_path_condition_numeric opposite_paths.
Proof.
  intros opposite_paths Hbool.
  unfold weak_axp_path_condition_numeric.
  apply opposite_path_blocking_equivalence.
  exact Hbool.
Qed.

Theorem numeric_weak_axp_path_condition_implies_boolean_blocking :
  forall opposite_paths : list Path,
    weak_axp_path_condition_numeric opposite_paths ->
    boolean_opposite_paths_blocked opposite_paths.
Proof.
  intros opposite_paths Hnum.
  apply opposite_path_blocking_equivalence.
  exact Hnum.
Qed.

(* ================================================================ *)
(* 11. Feature-subset partial assignment for weak AXp                 *)
(* ================================================================ *)

(* The previous section assumed that the opposite paths already contained
   the candidate partial-assignment constraints.

   This section models those constraints explicitly.

   Given:
     x0 : the explained instance,
     S  : the selected feature subset,

   numeric satisfiability under S means:
     there exists a valuation y such that y agrees exactly with x0 on
     every feature in S, and y satisfies the path.

   Boolean satisfiability under S means:
     there exists an order-consistent Boolean threshold assignment rho
     satisfying the encoded path, and rho agrees with x0 on every threshold
     atom whose feature belongs to S.

   The theorem below proves equivalence between the numeric and Boolean
   formulations for finite threshold paths.
*)

Fixpoint feature_in_bool (f : Feature) (S : list Feature) : bool :=
  match S with
  | [] => false
  | g :: tl =>
      if Nat.eqb f g
      then true
      else feature_in_bool f tl
  end.

Lemma feature_in_bool_in_true :
  forall (f : Feature) (S : list Feature),
    In f S ->
    feature_in_bool f S = true.
Proof.
  intros f S Hin.
  induction S as [| g tl IH].
  - contradiction.
  - simpl in *.
    destruct Hin as [Hhead | Htail].
    + subst g.
      rewrite Nat.eqb_refl.
      reflexivity.
    + destruct (Nat.eqb f g) eqn:E.
      * reflexivity.
      * apply IH. exact Htail.
Qed.

Lemma feature_in_bool_true_in :
  forall (f : Feature) (S : list Feature),
    feature_in_bool f S = true ->
    In f S.
Proof.
  intros f S H.
  induction S as [| g tl IH].
  - simpl in H. discriminate.
  - simpl in H.
    destruct (Nat.eqb f g) eqn:E.
    + apply Nat.eqb_eq in E.
      subst g.
      left. reflexivity.
    + right. apply IH. exact H.
Qed.

Definition fixed_on_selected
           (x0 y : Valuation) (S : list Feature) : Prop :=
  forall f : Feature,
    In f S ->
    y f = x0 f.

Definition selected_atom_agreement
           (x0 : Valuation)
           (S : list Feature)
           (atoms : list Atom)
           (rho : Assignment) : Prop :=
  forall (f : Feature) (t : Threshold),
    In (f,t) atoms ->
    In f S ->
    induced_assignment x0 (f,t) = rho (f,t).

Definition numeric_path_satisfiable_under_selection
           (x0 : Valuation) (S : list Feature) (path : Path) : Prop :=
  exists y : Valuation,
    fixed_on_selected x0 y S /\
    path_evalb y path = true.

Definition boolean_path_satisfiable_with_order_under_selection
           (x0 : Valuation) (S : list Feature) (path : Path) : Prop :=
  exists rho : Assignment,
    order_consistent_on (atoms_of_path path) rho /\
    selected_atom_agreement x0 S (atoms_of_path path) rho /\
    cnf_evalb rho (path_clauses path) = true.

Lemma induced_assignment_same_on_feature :
  forall (x0 y : Valuation) (f : Feature) (t : Threshold),
    y f = x0 f ->
    induced_assignment x0 (f,t) = induced_assignment y (f,t).
Proof.
  intros x0 y f t Hsame.
  unfold induced_assignment, atom_evalb.
  simpl.
  rewrite Hsame.
  reflexivity.
Qed.

Theorem numeric_selected_path_sat_implies_boolean_selected_path_sat :
  forall (path : Path) (x0 : Valuation) (S : list Feature),
    numeric_path_satisfiable_under_selection x0 S path ->
    boolean_path_satisfiable_with_order_under_selection x0 S path.
Proof.
  intros path x0 S [y [Hfixed Hpath]].
  exists (induced_assignment y).
  split.
  - apply induced_assignment_order_consistent.
  - split.
    + unfold selected_atom_agreement.
      intros f t HinAtom HinS.
      apply induced_assignment_same_on_feature.
      apply Hfixed.
      exact HinS.
    + rewrite path_encoding_correct.
      exact Hpath.
Qed.

Definition patched_valuation
           (x0 : Valuation)
           (atoms : list Atom)
           (rho : Assignment)
           (S : list Feature) : Valuation :=
  fun f =>
    if feature_in_bool f S
    then x0 f
    else canonical_valuation atoms rho f.

Lemma patched_valuation_fixed_on_selected :
  forall (x0 : Valuation) (atoms : list Atom)
         (rho : Assignment) (S : list Feature),
    fixed_on_selected x0 (patched_valuation x0 atoms rho S) S.
Proof.
  intros x0 atoms rho S.
  unfold fixed_on_selected.
  intros f Hin.
  unfold patched_valuation.
  rewrite (feature_in_bool_in_true f S Hin).
  reflexivity.
Qed.

Lemma patched_valuation_realizes_assignment :
  forall (atoms : list Atom) (rho : Assignment)
         (x0 : Valuation) (S : list Feature),
    order_consistent_on atoms rho ->
    selected_atom_agreement x0 S atoms rho ->
    forall (f : Feature) (t : Threshold),
      In (f,t) atoms ->
      induced_assignment (patched_valuation x0 atoms rho S) (f,t)
      = rho (f,t).
Proof.
  intros atoms rho x0 S Hord Hmatch f t HinAtom.
  unfold induced_assignment, atom_evalb.
  simpl.
  unfold patched_valuation.
  destruct (feature_in_bool f S) eqn:Eselected.
  - assert (HinS : In f S).
    {
      apply feature_in_bool_true_in.
      exact Eselected.
    }
    unfold selected_atom_agreement in Hmatch.
    specialize (Hmatch f t HinAtom HinS).
    unfold induced_assignment, atom_evalb in Hmatch.
    simpl in Hmatch.
    exact Hmatch.
  - change
      (atom_evalb (canonical_valuation atoms rho) (f,t)
       = rho (f,t)).
    apply canonical_valuation_realizes_assignment.
    + exact Hord.
    + exact HinAtom.
Qed.

Theorem boolean_selected_path_sat_implies_numeric_selected_path_sat :
  forall (path : Path) (x0 : Valuation) (S : list Feature),
    boolean_path_satisfiable_with_order_under_selection x0 S path ->
    numeric_path_satisfiable_under_selection x0 S path.
Proof.
  intros path x0 S [rho [Hord [Hmatch Hsat]]].
  exists (patched_valuation x0 (atoms_of_path path) rho S).
  split.
  - apply patched_valuation_fixed_on_selected.
  - rewrite <- path_encoding_correct.
    rewrite
      (path_clauses_assignment_agree
         path
         (induced_assignment
            (patched_valuation x0 (atoms_of_path path) rho S))
         rho).
    + exact Hsat.
    + intros [f t] HinAtom.
      apply patched_valuation_realizes_assignment.
      * exact Hord.
      * exact Hmatch.
      * exact HinAtom.
Qed.

Theorem selected_path_satisfiability_equivalence :
  forall (path : Path) (x0 : Valuation) (S : list Feature),
    numeric_path_satisfiable_under_selection x0 S path <->
    boolean_path_satisfiable_with_order_under_selection x0 S path.
Proof.
  intros path x0 S.
  split.
  - apply numeric_selected_path_sat_implies_boolean_selected_path_sat.
  - apply boolean_selected_path_sat_implies_numeric_selected_path_sat.
Qed.

Definition numeric_opposite_paths_blocked_under_selection
           (x0 : Valuation) (S : list Feature)
           (opposite_paths : list Path) : Prop :=
  forall p : Path,
    In p opposite_paths ->
    ~ numeric_path_satisfiable_under_selection x0 S p.

Definition boolean_opposite_paths_blocked_under_selection
           (x0 : Valuation) (S : list Feature)
           (opposite_paths : list Path) : Prop :=
  forall p : Path,
    In p opposite_paths ->
    ~ boolean_path_satisfiable_with_order_under_selection x0 S p.

Theorem selected_opposite_path_blocking_equivalence :
  forall (x0 : Valuation) (S : list Feature)
         (opposite_paths : list Path),
    numeric_opposite_paths_blocked_under_selection x0 S opposite_paths <->
    boolean_opposite_paths_blocked_under_selection x0 S opposite_paths.
Proof.
  intros x0 S opposite_paths.
  split.
  - intros Hnum p Hin Hboolsat.
    apply (Hnum p Hin).
    apply boolean_selected_path_sat_implies_numeric_selected_path_sat.
    exact Hboolsat.
  - intros Hbool p Hin Hnumsat.
    apply (Hbool p Hin).
    apply numeric_selected_path_sat_implies_boolean_selected_path_sat.
    exact Hnumsat.
Qed.

Definition weak_axp_selected_path_condition_numeric
           (x0 : Valuation) (S : list Feature)
           (opposite_paths : list Path) : Prop :=
  numeric_opposite_paths_blocked_under_selection x0 S opposite_paths.

Definition weak_axp_selected_path_condition_boolean
           (x0 : Valuation) (S : list Feature)
           (opposite_paths : list Path) : Prop :=
  boolean_opposite_paths_blocked_under_selection x0 S opposite_paths.

Theorem weak_axp_selected_path_condition_equivalence :
  forall (x0 : Valuation) (S : list Feature)
         (opposite_paths : list Path),
    weak_axp_selected_path_condition_numeric x0 S opposite_paths <->
    weak_axp_selected_path_condition_boolean x0 S opposite_paths.
Proof.
  intros x0 S opposite_paths.
  unfold weak_axp_selected_path_condition_numeric.
  unfold weak_axp_selected_path_condition_boolean.
  apply selected_opposite_path_blocking_equivalence.
Qed.

(* ================================================================ *)
(* 12. Deletion-based AXp extraction minimality                       *)
(* ================================================================ *)

(* This section verifies the abstract deletion procedure used by many
   explanation algorithms.

   We model a candidate explanation as a finite list of selected features.
   A Boolean predicate P : list Feature -> bool represents the already-proved
   weak-AXp test.

   Assumption:
     P is monotone upward with respect to set inclusion:
       if A ⊆ B and P A = true, then P B = true.

   This matches weak AXp intuition:
     fixing more features cannot create a new opposite-class counterexample
     if none existed before.

   The deletion algorithm scans features and removes a feature whenever P
   remains true without it.

   Main theorem:
     deletion_extraction_returns_subset_minimal_axp

   Meaning:
     assuming P initial = true and P is monotone, the deletion algorithm
     returns a subset-minimal set satisfying P.
*)

Definition subset_features (A B : list Feature) : Prop :=
  forall f : Feature, In f A -> In f B.

Definition monotone_feature_predicate
           (P : list Feature -> bool) : Prop :=
  forall A B : list Feature,
    subset_features A B ->
    P A = true ->
    P B = true.

Fixpoint remove_feature (f : Feature) (S : list Feature) : list Feature :=
  match S with
  | [] => []
  | g :: tl =>
      if Nat.eqb f g
      then remove_feature f tl
      else g :: remove_feature f tl
  end.

Lemma subset_features_refl :
  forall S : list Feature,
    subset_features S S.
Proof.
  intros S f Hin.
  exact Hin.
Qed.

Lemma subset_features_trans :
  forall A B C : list Feature,
    subset_features A B ->
    subset_features B C ->
    subset_features A C.
Proof.
  intros A B C HAB HBC f Hin.
  apply HBC.
  apply HAB.
  exact Hin.
Qed.

Lemma remove_feature_no_in :
  forall (f : Feature) (S : list Feature),
    ~ In f (remove_feature f S).
Proof.
  intros f S.
  induction S as [| g tl IH].
  - simpl. intro H. exact H.
  - simpl.
    destruct (Nat.eqb f g) eqn:E.
    + exact IH.
    + intro H.
      simpl in H.
      destruct H as [Hhead | Htail].
      * subst g.
        rewrite Nat.eqb_refl in E.
        discriminate.
      * apply IH. exact Htail.
Qed.

Lemma remove_feature_subset :
  forall (f : Feature) (S : list Feature),
    subset_features (remove_feature f S) S.
Proof.
  intros f S h Hin.
  induction S as [| g tl IH].
  - simpl in Hin. contradiction.
  - simpl in Hin.
    destruct (Nat.eqb f g) eqn:E.
    + right.
      apply IH.
      exact Hin.
    + simpl in Hin.
      destruct Hin as [Hhead | Htail].
      * left. exact Hhead.
      * right.
        apply IH.
        exact Htail.
Qed.

Lemma in_remove_feature_intro :
  forall (f h : Feature) (S : list Feature),
    In h S ->
    h <> f ->
    In h (remove_feature f S).
Proof.
  intros f h S Hin Hneq.
  induction S as [| g tl IH].
  - contradiction.
  - simpl in *.
    destruct Hin as [Hhead | Htail].
    + subst g.
      destruct (Nat.eqb f h) eqn:E.
      * apply Nat.eqb_eq in E.
        symmetry in E.
        contradiction.
      * simpl. left. reflexivity.
    + destruct (Nat.eqb f g) eqn:E.
      * apply IH. exact Htail.
      * simpl. right.
        apply IH. exact Htail.
Qed.

Lemma remove_feature_mono :
  forall (f : Feature) (A B : list Feature),
    subset_features A B ->
    subset_features (remove_feature f A) (remove_feature f B).
Proof.
  intros f A B Hsub h Hin.
  apply in_remove_feature_intro.
  - apply Hsub.
    eapply remove_feature_subset.
    exact Hin.
  - intro Heq.
    subst h.
    apply (remove_feature_no_in f A).
    exact Hin.
Qed.

Fixpoint deletion_extract
         (P : list Feature -> bool)
         (todo current : list Feature) : list Feature :=
  match todo with
  | [] => current
  | f :: tl =>
      let without_f := remove_feature f current in
      if P without_f
      then deletion_extract P tl without_f
      else deletion_extract P tl current
  end.

Definition axp_extraction
           (P : list Feature -> bool)
           (initial : list Feature) : list Feature :=
  deletion_extract P initial initial.

Lemma deletion_extract_preserves_true :
  forall (P : list Feature -> bool)
         (todo current : list Feature),
    P current = true ->
    P (deletion_extract P todo current) = true.
Proof.
  intros P todo.
  induction todo as [| f tl IH]; intros current Hcurrent.
  - simpl. exact Hcurrent.
  - simpl.
    destruct (P (remove_feature f current)) eqn:E.
    + apply IH. exact E.
    + apply IH. exact Hcurrent.
Qed.

Lemma deletion_extract_subset_current :
  forall (P : list Feature -> bool)
         (todo current : list Feature),
    subset_features (deletion_extract P todo current) current.
Proof.
  intros P todo.
  induction todo as [| f tl IH]; intros current.
  - simpl. apply subset_features_refl.
  - simpl.
    destruct (P (remove_feature f current)) eqn:E.
    + eapply subset_features_trans.
      * apply IH.
      * apply remove_feature_subset.
    + apply IH.
Qed.

Lemma deletion_extract_processed_minimal :
  forall (P : list Feature -> bool),
    monotone_feature_predicate P ->
    forall (todo current : list Feature),
      P current = true ->
      forall f : Feature,
        In f todo ->
        In f (deletion_extract P todo current) ->
        P (remove_feature f (deletion_extract P todo current)) = false.
Proof.
  intros P Hmono todo.
  induction todo as [| h tl IH]; intros current Hcurrent f HinTodo HinFinal.
  - simpl in HinTodo. contradiction.
  - simpl in *.
    destruct (P (remove_feature h current)) eqn:Eremove.
    + destruct HinTodo as [Hfh | Hftl].
      * subst f.
        assert
          (Hsub :
             subset_features
               (deletion_extract P tl (remove_feature h current))
               (remove_feature h current)).
        {
          apply deletion_extract_subset_current.
        }
        exfalso.
        apply (remove_feature_no_in h current).
        apply Hsub.
        exact HinFinal.
      * apply IH.
        -- exact Eremove.
        -- exact Hftl.
        -- exact HinFinal.
    + destruct HinTodo as [Hfh | Hftl].
      * subst f.
        destruct
          (P (remove_feature h (deletion_extract P tl current)))
          eqn:Efinal.
        -- exfalso.
           assert
             (HsubFinal :
                subset_features (deletion_extract P tl current) current).
           {
             apply deletion_extract_subset_current.
           }
           assert
             (HsubRemove :
                subset_features
                  (remove_feature h (deletion_extract P tl current))
                  (remove_feature h current)).
           {
             apply remove_feature_mono.
             exact HsubFinal.
           }
           specialize
             (Hmono
                (remove_feature h (deletion_extract P tl current))
                (remove_feature h current)
                HsubRemove
                Efinal).
           rewrite Eremove in Hmono.
           discriminate.
        -- reflexivity.
      * apply IH.
        -- exact Hcurrent.
        -- exact Hftl.
        -- exact HinFinal.
Qed.

Definition deletion_minimal_axp_bool
           (P : list Feature -> bool)
           (S : list Feature) : Prop :=
  P S = true /\
  forall f : Feature,
    In f S ->
    P (remove_feature f S) = false.

Definition subset_minimal_axp_bool
           (P : list Feature -> bool)
           (S : list Feature) : Prop :=
  P S = true /\
  forall T : list Feature,
    subset_features T S ->
    P T = true ->
    subset_features S T.

Theorem deletion_extraction_returns_deletion_minimal_axp :
  forall (P : list Feature -> bool) (initial : list Feature),
    monotone_feature_predicate P ->
    P initial = true ->
    deletion_minimal_axp_bool P (axp_extraction P initial).
Proof.
  intros P initial Hmono Hinitial.
  unfold deletion_minimal_axp_bool.
  unfold axp_extraction.
  split.
  - apply deletion_extract_preserves_true.
    exact Hinitial.
  - intros f HinFinal.
    apply deletion_extract_processed_minimal with
      (todo := initial)
      (current := initial).
    + exact Hmono.
    + exact Hinitial.
    + assert
        (Hsub :
           subset_features (deletion_extract P initial initial) initial).
      {
        apply deletion_extract_subset_current.
      }
      apply Hsub.
      exact HinFinal.
    + exact HinFinal.
Qed.

Theorem deletion_minimal_implies_subset_minimal :
  forall (P : list Feature -> bool) (S : list Feature),
    monotone_feature_predicate P ->
    deletion_minimal_axp_bool P S ->
    subset_minimal_axp_bool P S.
Proof.
  intros P S Hmono [HPS Hdel].
  unfold subset_minimal_axp_bool.
  split.
  - exact HPS.
  - intros T HTS HPT.
    unfold subset_features.
    intros f HinS.
    destruct (in_dec Nat.eq_dec f T) as [HinT | HnotInT].
    + exact HinT.
    + exfalso.
      assert (HT_remove : subset_features T (remove_feature f S)).
      {
        intros h HinHT.
        apply in_remove_feature_intro.
        - apply HTS. exact HinHT.
        - intro Heq.
          subst h.
          contradiction.
      }
      assert (HremoveTrue : P (remove_feature f S) = true).
      {
        apply Hmono with (A := T).
        - exact HT_remove.
        - exact HPT.
      }
      rewrite (Hdel f HinS) in HremoveTrue.
      discriminate.
Qed.

Theorem deletion_extraction_returns_subset_minimal_axp :
  forall (P : list Feature -> bool) (initial : list Feature),
    monotone_feature_predicate P ->
    P initial = true ->
    subset_minimal_axp_bool P (axp_extraction P initial).
Proof.
  intros P initial Hmono Hinitial.
  apply deletion_minimal_implies_subset_minimal.
  - exact Hmono.
  - apply deletion_extraction_returns_deletion_minimal_axp.
    + exact Hmono.
    + exact Hinitial.
Qed.

(* ================================================================ *)
(* 13. Monotonicity of the selected weak-AXp condition                *)
(* ================================================================ *)

(* The deletion theorem above is abstract: it assumes that the predicate P
   used by the deletion algorithm is monotone.

   This section proves that the selected-feature weak-AXp blocking condition
   is monotone.

   If A ⊆ B and all opposite paths are blocked while fixing A, then all
   opposite paths are also blocked while fixing B.

   Reason:
     fixing more features only adds constraints. Any counterexample satisfying
     the larger fixed set B would also satisfy the smaller fixed set A.
*)

Lemma selected_atom_agreement_weaken :
  forall (x0 : Valuation) (A B : list Feature)
         (atoms : list Atom) (rho : Assignment),
    subset_features A B ->
    selected_atom_agreement x0 B atoms rho ->
    selected_atom_agreement x0 A atoms rho.
Proof.
  intros x0 A B atoms rho HAB HagreeB.
  unfold selected_atom_agreement in *.
  intros f t HinAtom HinA.
  apply HagreeB.
  - exact HinAtom.
  - apply HAB. exact HinA.
Qed.

Lemma boolean_selected_path_sat_weaken :
  forall (path : Path) (x0 : Valuation)
         (A B : list Feature),
    subset_features A B ->
    boolean_path_satisfiable_with_order_under_selection x0 B path ->
    boolean_path_satisfiable_with_order_under_selection x0 A path.
Proof.
  intros path x0 A B HAB [rho [Hord [HagreeB Hsat]]].
  exists rho.
  split.
  - exact Hord.
  - split.
    + apply selected_atom_agreement_weaken with (B := B).
      * exact HAB.
      * exact HagreeB.
    + exact Hsat.
Qed.

Theorem weak_axp_selected_path_condition_boolean_monotone :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (A B : list Feature),
    subset_features A B ->
    weak_axp_selected_path_condition_boolean x0 A opposite_paths ->
    weak_axp_selected_path_condition_boolean x0 B opposite_paths.
Proof.
  intros x0 opposite_paths A B HAB HblockedA.
  unfold weak_axp_selected_path_condition_boolean in *.
  unfold boolean_opposite_paths_blocked_under_selection in *.
  intros p HinOpp HsatB.
  apply (HblockedA p HinOpp).
  apply boolean_selected_path_sat_weaken with (B := B).
  - exact HAB.
  - exact HsatB.
Qed.

Theorem weak_axp_selected_path_condition_numeric_monotone :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (A B : list Feature),
    subset_features A B ->
    weak_axp_selected_path_condition_numeric x0 A opposite_paths ->
    weak_axp_selected_path_condition_numeric x0 B opposite_paths.
Proof.
  intros x0 opposite_paths A B HAB HnumA.
  apply weak_axp_selected_path_condition_equivalence.
  apply weak_axp_selected_path_condition_boolean_monotone with (A := A).
  - exact HAB.
  - apply weak_axp_selected_path_condition_equivalence.
    exact HnumA.
Qed.

(* Bridge from the Prop-level weak-AXp condition to a Boolean checker.

   A real implementation usually exposes a Boolean function P.
   If P is proven equivalent to the Coq weak-AXp condition below, then P is
   monotone and can be used safely by the verified deletion extractor.
*)

Definition bool_reflects_weak_axp_selected_condition
           (P : list Feature -> bool)
           (x0 : Valuation)
           (opposite_paths : list Path) : Prop :=
  forall S : list Feature,
    P S = true <->
    weak_axp_selected_path_condition_boolean x0 S opposite_paths.

Theorem reflected_weak_axp_predicate_is_monotone :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (P : list Feature -> bool),
    bool_reflects_weak_axp_selected_condition P x0 opposite_paths ->
    monotone_feature_predicate P.
Proof.
  intros x0 opposite_paths P Hreflect.
  unfold monotone_feature_predicate.
  intros A B HAB HPA.
  destruct (Hreflect A) as [HPA_to_prop _].
  destruct (Hreflect B) as [_ Hprop_to_PB].
  apply Hprop_to_PB.
  apply weak_axp_selected_path_condition_boolean_monotone with (A := A).
  - exact HAB.
  - apply HPA_to_prop.
    exact HPA.
Qed.

Definition subset_minimal_weak_axp_selected_condition_boolean
           (x0 : Valuation)
           (opposite_paths : list Path)
           (S : list Feature) : Prop :=
  weak_axp_selected_path_condition_boolean x0 S opposite_paths /\
  forall T : list Feature,
    subset_features T S ->
    weak_axp_selected_path_condition_boolean x0 T opposite_paths ->
    subset_features S T.

Theorem axp_extraction_returns_subset_minimal_weak_axp_selected_condition :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (P : list Feature -> bool) (initial : list Feature),
    bool_reflects_weak_axp_selected_condition P x0 opposite_paths ->
    P initial = true ->
    subset_minimal_weak_axp_selected_condition_boolean
      x0 opposite_paths (axp_extraction P initial).
Proof.
  intros x0 opposite_paths P initial Hreflect Hinitial.
  assert (Hmono : monotone_feature_predicate P).
  {
    apply reflected_weak_axp_predicate_is_monotone
      with (x0 := x0) (opposite_paths := opposite_paths).
    exact Hreflect.
  }
  pose proof
    (deletion_extraction_returns_subset_minimal_axp
       P initial Hmono Hinitial)
    as HminP.
  unfold subset_minimal_axp_bool in HminP.
  destruct HminP as [HPresult HminimalP].
  unfold subset_minimal_weak_axp_selected_condition_boolean.
  split.
  - destruct (Hreflect (axp_extraction P initial)) as [HP_to_prop _].
    apply HP_to_prop.
    exact HPresult.
  - intros T HTsub HTweak.
    apply HminimalP.
    + exact HTsub.
    + destruct (Hreflect T) as [_ Hprop_to_P].
      apply Hprop_to_P.
      exact HTweak.
Qed.

(* ================================================================ *)
(* 14. Certified tree-language compliance                             *)
(* ================================================================ *)


(* Replace your current Section 14 by this version.  This version uses
   explicit theorem applications instead of relying on Coq's inference in
   rewrite/apply steps. *)

Inductive DeclaredFamily : Type :=
| FamilyHorn
| FamilyAntiHorn
| FamilyAny.

Definition predicate_in_family_bool
           (family : DeclaredFamily) (p : Predicate) : bool :=
  match family with
  | FamilyHorn => is_horn_predicate p
  | FamilyAntiHorn => is_antihorn_predicate p
  | FamilyAny => true
  end.

Definition predicate_in_family
           (family : DeclaredFamily) (p : Predicate) : Prop :=
  predicate_in_family_bool family p = true.

Definition Label := nat.

Inductive MDT : Type :=
| MDTLeaf : Label -> MDT
| MDTNode : Predicate -> MDT -> MDT -> MDT.


Fixpoint mdt_respects_family_bool
         (family : DeclaredFamily) (tree : MDT) : bool :=
  match tree with
  | MDTLeaf _ => true
  | MDTNode p left_tree right_tree =>
      andb
        (predicate_in_family_bool family p)
        (andb
           (mdt_respects_family_bool family left_tree)
           (mdt_respects_family_bool family right_tree))
  end.

Fixpoint mdt_respects_family
         (family : DeclaredFamily) (tree : MDT) : Prop :=
  match tree with
  | MDTLeaf _ => True
  | MDTNode p left_tree right_tree =>
      predicate_in_family family p /\
      mdt_respects_family family left_tree /\
      mdt_respects_family family right_tree
  end.
  
Theorem mdt_language_checker_sound :
  forall (family : DeclaredFamily) (tree : MDT),
    mdt_respects_family_bool family tree = true ->
    mdt_respects_family family tree.
Proof.
  intros family tree.
  induction tree as [label | p left IHleft right IHright].
  - simpl. intros _. exact I.
  - simpl. intros H.
    apply andb_true_iff in H.
    destruct H as [Hp Hchildren].
    apply andb_true_iff in Hchildren.
    destruct Hchildren as [Hleft Hright].
    split.
    + unfold predicate_in_family.
      exact Hp.
    + split.
      * exact (IHleft Hleft).
      * exact (IHright Hright).
Qed.

Theorem mdt_language_checker_complete :
  forall (family : DeclaredFamily) (tree : MDT),
    mdt_respects_family family tree ->
    mdt_respects_family_bool family tree = true.
Proof.
  intros family tree.
  induction tree as [label | p left IHleft right IHright].
  - simpl. intros _. reflexivity.
  - simpl. intros H.
    destruct H as [Hp [Hleft Hright]].
    apply andb_true_iff.
    split.
    + unfold predicate_in_family in Hp.
      exact Hp.
    + apply andb_true_iff.
      split.
      * exact (IHleft Hleft).
      * exact (IHright Hright).
Qed.

Fixpoint paths_of_mdt (tree : MDT) : list Path :=
  match tree with
  | MDTLeaf _ => [[]]
  | MDTNode p left_tree right_tree =>
      map (fun path => (p, true) :: path) (paths_of_mdt left_tree)
      ++
      map (fun path => (p, false) :: path) (paths_of_mdt right_tree)
  end.

Fixpoint path_predicates_in_family
         (family : DeclaredFamily) (path : Path) : Prop :=
  match path with
  | [] => True
  | (p, _) :: tl =>
      predicate_in_family family p /\
      path_predicates_in_family family tl
  end.

Theorem mdt_respects_family_paths :
  forall (family : DeclaredFamily) (tree : MDT) (path : Path),
    mdt_respects_family family tree ->
    In path (paths_of_mdt tree) ->
    path_predicates_in_family family path.
Proof.
  intros family tree.
  induction tree as [label | p left_tree IHleft right_tree IHright];
    intros path Htree Hin.
  - simpl in Hin.
    destruct Hin as [Hpath | Hbad].
    + subst path. simpl. exact I.
    + contradiction.
  - simpl in Htree.
    destruct Htree as [Hp [Hleft Hright]].
    simpl in Hin.
    apply in_app_or in Hin.
    destruct Hin as [HinLeft | HinRight].
    + apply in_map_iff in HinLeft.
      destruct HinLeft as [subpath [Heq HinSub]].
      subst path.
      simpl.
      split.
      * exact Hp.
      * exact (IHleft subpath Hleft HinSub).
    + apply in_map_iff in HinRight.
      destruct HinRight as [subpath [Heq HinSub]].
      subst path.
      simpl.
      split.
      * exact Hp.
      * exact (IHright subpath Hright HinSub).
Qed.

Lemma all_horn_clauses_app :
  forall a b : CNF,
    all_horn_clauses (a ++ b)
    =
    andb (all_horn_clauses a) (all_horn_clauses b).
Proof.
  intros a b.
  unfold all_horn_clauses.
  rewrite forallb_app.
  reflexivity.
Qed.

Lemma all_antihorn_clauses_app :
  forall a b : CNF,
    all_antihorn_clauses (a ++ b)
    =
    andb (all_antihorn_clauses a) (all_antihorn_clauses b).
Proof.
  intros a b.
  unfold all_antihorn_clauses.
  rewrite forallb_app.
  reflexivity.
Qed.

Lemma horn_edge_clauses_are_horn :
  forall (p : Predicate) (branch : bool),
    predicate_in_family FamilyHorn p ->
    all_horn_clauses (edge_clauses (p, branch)) = true.
Proof.
  intros p branch Hp.
  unfold predicate_in_family in Hp.
  simpl in Hp.
  destruct branch.
  - simpl.
    unfold all_horn_clauses.
    simpl.
    rewrite (horn_predicate_true_branch_is_horn p Hp).
    reflexivity.
  - simpl.
    exact (false_branch_clauses_are_horn p).
Qed.

Lemma antihorn_edge_clauses_are_antihorn :
  forall (p : Predicate) (branch : bool),
    predicate_in_family FamilyAntiHorn p ->
    all_antihorn_clauses (edge_clauses (p, branch)) = true.
Proof.
  intros p branch Hp.
  unfold predicate_in_family in Hp.
  simpl in Hp.
  destruct branch.
  - simpl.
    unfold all_antihorn_clauses.
    simpl.
    rewrite (antihorn_predicate_true_branch_is_antihorn p Hp).
    reflexivity.
  - simpl.
    exact (false_branch_clauses_are_antihorn p).
Qed.

Theorem horn_path_clauses_are_horn :
  forall path : Path,
    path_predicates_in_family FamilyHorn path ->
    all_horn_clauses (path_clauses path) = true.
Proof.
  intros path.
  induction path as [| [p branch] tl IH].
  - simpl. reflexivity.
  - simpl.
    intros Hfam.
    destruct Hfam as [Hp Htl].
    change
      (all_horn_clauses
         (edge_clauses (p, branch) ++ path_clauses tl) = true).
    rewrite
      (all_horn_clauses_app
         (edge_clauses (p, branch))
         (path_clauses tl)).
    apply andb_true_iff.
    split.
    + exact (horn_edge_clauses_are_horn p branch Hp).
    + exact (IH Htl).
Qed.

Theorem antihorn_path_clauses_are_antihorn :
  forall path : Path,
    path_predicates_in_family FamilyAntiHorn path ->
    all_antihorn_clauses (path_clauses path) = true.
Proof.
  intros path.
  induction path as [| [p branch] tl IH].
  - simpl. reflexivity.
  - simpl.
    intros Hfam.
    destruct Hfam as [Hp Htl].
    change
      (all_antihorn_clauses
         (edge_clauses (p, branch) ++ path_clauses tl) = true).
    rewrite
      (all_antihorn_clauses_app
         (edge_clauses (p, branch))
         (path_clauses tl)).
    apply andb_true_iff.
    split.
    + exact (antihorn_edge_clauses_are_antihorn p branch Hp).
    + exact (IH Htl).
Qed.

Theorem checked_horn_mdt_paths_have_horn_cnf :
  forall (tree : MDT) (path : Path),
    mdt_respects_family_bool FamilyHorn tree = true ->
    In path (paths_of_mdt tree) ->
    all_horn_clauses (path_clauses path) = true.
Proof.
  intros tree path Hcheck HinPath.
  apply horn_path_clauses_are_horn.
  exact
    (mdt_respects_family_paths
       FamilyHorn
       tree
       path
       (mdt_language_checker_sound FamilyHorn tree Hcheck)
       HinPath).
Qed.

Theorem checked_antihorn_mdt_paths_have_antihorn_cnf :
  forall (tree : MDT) (path : Path),
    mdt_respects_family_bool FamilyAntiHorn tree = true ->
    In path (paths_of_mdt tree) ->
    all_antihorn_clauses (path_clauses path) = true.
Proof.
  intros tree path Hcheck HinPath.
  apply antihorn_path_clauses_are_antihorn.
  exact
    (mdt_respects_family_paths
       FamilyAntiHorn
       tree
       path
       (mdt_language_checker_sound FamilyAntiHorn tree Hcheck)
       HinPath).
Qed.

(* ================================================================ *)
(* 15. Small concrete sanity examples                                 *)
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

(* ================================================================ *)
(* 15. Certified SAT-solver answer layer                              *)
(* ================================================================ *)

(* This section starts the solver-correctness/refinement layer.

   It does not yet prove a full Horn unit-propagation implementation.
   Instead, it proves a certified answer checker:

     - SAT answers are accepted only when a concrete finite model certificate
       really satisfies the CNF.

     - UNSAT-by-empty-clause answers are accepted only when the CNF really is
       unsatisfiable.

   This is the first safe bridge toward Python refinement:
   a Python Horn/AntiHorn solver can export one of these certificates, and Coq
   checks the certificate instead of trusting Python directly.
*)

Definition atom_eqb (a b : Atom) : bool :=
  match a, b with
  | (f,t), (g,u) =>
      andb (Nat.eqb f g) (Qeq_bool t u)
  end.

Fixpoint atom_in_bool (a : Atom) (atoms : list Atom) : bool :=
  match atoms with
  | [] => false
  | b :: tl =>
      if atom_eqb a b
      then true
      else atom_in_bool a tl
  end.

Definition finite_assignment (true_atoms : list Atom) : Assignment :=
  fun a => atom_in_bool a true_atoms.

Definition sat_model_certificate_check
           (cnf : CNF) (true_atoms : list Atom) : bool :=
  cnf_evalb (finite_assignment true_atoms) cnf.

Theorem sat_model_certificate_sound :
  forall (cnf : CNF) (true_atoms : list Atom),
    sat_model_certificate_check cnf true_atoms = true ->
    exists rho : Assignment,
      cnf_evalb rho cnf = true.
Proof.
  intros cnf true_atoms Hcheck.
  exists (finite_assignment true_atoms).
  unfold sat_model_certificate_check in Hcheck.
  exact Hcheck.
Qed.

Definition horn_sat_certificate_check
           (cnf : CNF) (true_atoms : list Atom) : bool :=
  andb
    (all_horn_clauses cnf)
    (sat_model_certificate_check cnf true_atoms).

Definition antihorn_sat_certificate_check
           (cnf : CNF) (true_atoms : list Atom) : bool :=
  andb
    (all_antihorn_clauses cnf)
    (sat_model_certificate_check cnf true_atoms).

Theorem horn_sat_certificate_sound :
  forall (cnf : CNF) (true_atoms : list Atom),
    horn_sat_certificate_check cnf true_atoms = true ->
    all_horn_clauses cnf = true /\
    exists rho : Assignment,
      cnf_evalb rho cnf = true.
Proof.
  intros cnf true_atoms Hcheck.
  unfold horn_sat_certificate_check in Hcheck.
  apply andb_true_iff in Hcheck.
  destruct Hcheck as [Hhorn Hsat].
  split.
  - exact Hhorn.
  - apply sat_model_certificate_sound with (true_atoms := true_atoms).
    exact Hsat.
Qed.

Theorem antihorn_sat_certificate_sound :
  forall (cnf : CNF) (true_atoms : list Atom),
    antihorn_sat_certificate_check cnf true_atoms = true ->
    all_antihorn_clauses cnf = true /\
    exists rho : Assignment,
      cnf_evalb rho cnf = true.
Proof.
  intros cnf true_atoms Hcheck.
  unfold antihorn_sat_certificate_check in Hcheck.
  apply andb_true_iff in Hcheck.
  destruct Hcheck as [Hantihorn Hsat].
  split.
  - exact Hantihorn.
  - apply sat_model_certificate_sound with (true_atoms := true_atoms).
    exact Hsat.
Qed.

Definition is_empty_clause (c : Clause) : bool :=
  match c with
  | [] => true
  | _ :: _ => false
  end.

Definition empty_clause_unsat_certificate_check (cnf : CNF) : bool :=
  existsb is_empty_clause cnf.

Lemma empty_clause_in_cnf_unsat :
  forall cnf : CNF,
    In [] cnf ->
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf Hempty.
  intros [rho Hsat].
  unfold cnf_evalb in Hsat.
  apply forallb_forall with (x := []) in Hsat.
  - simpl in Hsat.
    discriminate.
  - exact Hempty.
Qed.

Theorem empty_clause_unsat_certificate_sound :
  forall cnf : CNF,
    empty_clause_unsat_certificate_check cnf = true ->
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf Hcheck.
  unfold empty_clause_unsat_certificate_check in Hcheck.
  apply existsb_exists in Hcheck.
  destruct Hcheck as [c [Hin Hempty]].
  destruct c as [| s tl].
  - apply empty_clause_in_cnf_unsat.
    exact Hin.
  - simpl in Hempty.
    discriminate.
Qed.

Definition horn_empty_unsat_certificate_check (cnf : CNF) : bool :=
  andb
    (all_horn_clauses cnf)
    (empty_clause_unsat_certificate_check cnf).

Definition antihorn_empty_unsat_certificate_check (cnf : CNF) : bool :=
  andb
    (all_antihorn_clauses cnf)
    (empty_clause_unsat_certificate_check cnf).

Theorem horn_empty_unsat_certificate_sound :
  forall cnf : CNF,
    horn_empty_unsat_certificate_check cnf = true ->
    all_horn_clauses cnf = true /\
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf Hcheck.
  unfold horn_empty_unsat_certificate_check in Hcheck.
  apply andb_true_iff in Hcheck.
  destruct Hcheck as [Hhorn Hunsat].
  split.
  - exact Hhorn.
  - apply empty_clause_unsat_certificate_sound.
    exact Hunsat.
Qed.

Theorem antihorn_empty_unsat_certificate_sound :
  forall cnf : CNF,
    antihorn_empty_unsat_certificate_check cnf = true ->
    all_antihorn_clauses cnf = true /\
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf Hcheck.
  unfold antihorn_empty_unsat_certificate_check in Hcheck.
  apply andb_true_iff in Hcheck.
  destruct Hcheck as [Hantihorn Hunsat].
  split.
  - exact Hantihorn.
  - apply empty_clause_unsat_certificate_sound.
    exact Hunsat.
Qed.

Inductive CertifiedSolverAnswer : Type :=
| CertifiedSat : list Atom -> CertifiedSolverAnswer
| CertifiedUnsatByEmptyClause : CertifiedSolverAnswer
| CertifiedUnknown : CertifiedSolverAnswer.

Definition certified_horn_solver_answer_check
           (cnf : CNF) (ans : CertifiedSolverAnswer) : bool :=
  match ans with
  | CertifiedSat true_atoms =>
      horn_sat_certificate_check cnf true_atoms
  | CertifiedUnsatByEmptyClause =>
      horn_empty_unsat_certificate_check cnf
  | CertifiedUnknown =>
      true
  end.

Definition certified_antihorn_solver_answer_check
           (cnf : CNF) (ans : CertifiedSolverAnswer) : bool :=
  match ans with
  | CertifiedSat true_atoms =>
      antihorn_sat_certificate_check cnf true_atoms
  | CertifiedUnsatByEmptyClause =>
      antihorn_empty_unsat_certificate_check cnf
  | CertifiedUnknown =>
      true
  end.

Definition certified_horn_solver_answer_sound
           (cnf : CNF) (ans : CertifiedSolverAnswer) : Prop :=
  match ans with
  | CertifiedSat _ =>
      all_horn_clauses cnf = true /\
      exists rho : Assignment,
        cnf_evalb rho cnf = true
  | CertifiedUnsatByEmptyClause =>
      all_horn_clauses cnf = true /\
      ~ exists rho : Assignment,
          cnf_evalb rho cnf = true
  | CertifiedUnknown =>
      True
  end.

Definition certified_antihorn_solver_answer_sound
           (cnf : CNF) (ans : CertifiedSolverAnswer) : Prop :=
  match ans with
  | CertifiedSat _ =>
      all_antihorn_clauses cnf = true /\
      exists rho : Assignment,
        cnf_evalb rho cnf = true
  | CertifiedUnsatByEmptyClause =>
      all_antihorn_clauses cnf = true /\
      ~ exists rho : Assignment,
          cnf_evalb rho cnf = true
  | CertifiedUnknown =>
      True
  end.

Theorem certified_horn_solver_answer_checker_sound :
  forall (cnf : CNF) (ans : CertifiedSolverAnswer),
    certified_horn_solver_answer_check cnf ans = true ->
    certified_horn_solver_answer_sound cnf ans.
Proof.
  intros cnf ans Hcheck.
  destruct ans as [true_atoms | |].
  - simpl in *.
    apply horn_sat_certificate_sound with (true_atoms := true_atoms).
    exact Hcheck.
  - simpl in *.
    apply horn_empty_unsat_certificate_sound.
    exact Hcheck.
  - simpl. exact I.
Qed.

Theorem certified_antihorn_solver_answer_checker_sound :
  forall (cnf : CNF) (ans : CertifiedSolverAnswer),
    certified_antihorn_solver_answer_check cnf ans = true ->
    certified_antihorn_solver_answer_sound cnf ans.
Proof.
  intros cnf ans Hcheck.
  destruct ans as [true_atoms | |].
  - simpl in *.
    apply antihorn_sat_certificate_sound with (true_atoms := true_atoms).
    exact Hcheck.
  - simpl in *.
    apply antihorn_empty_unsat_certificate_sound.
    exact Hcheck.
  - simpl. exact I.
Qed.

(* ================================================================ *)
(* 16. Verified finite exhaustive CNF solver layer                    *)
(* ================================================================ *)

(* This section proves correctness of a concrete finite solver skeleton.

   The solver receives a finite list of candidate assignments.  Each candidate
   is represented by the list of atoms assigned true; all other atoms are
   assigned false by finite_assignment.

   The solver scans the candidates and returns the first one satisfying the
   CNF.

   What is proved here:
     - if the solver returns Some model, the model really satisfies the CNF;
     - if the solver returns None and the candidate list is complete for the
       CNF, then the CNF is genuinely unsatisfiable;
     - the same results are wrapped for Horn and AntiHorn CNFs.

   The next section can prove that a generated powerset of all CNF atoms is a
   complete candidate list. *)

Fixpoint find_satisfying_assignment
         (cnf : CNF) (candidates : list (list Atom))
  : option (list Atom) :=
  match candidates with
  | [] => None
  | true_atoms :: tl =>
      if sat_model_certificate_check cnf true_atoms
      then Some true_atoms
      else find_satisfying_assignment cnf tl
  end.

Theorem find_satisfying_assignment_sound :
  forall (cnf : CNF) (candidates : list (list Atom))
         (true_atoms : list Atom),
    find_satisfying_assignment cnf candidates = Some true_atoms ->
    cnf_evalb (finite_assignment true_atoms) cnf = true.
Proof.
  intros cnf candidates.
  induction candidates as [| cand tl IH]; intros true_atoms Hfind.
  - simpl in Hfind. discriminate.
  - simpl in Hfind.
    destruct (sat_model_certificate_check cnf cand) eqn:Hcand.
    + inversion Hfind; subst; clear Hfind.
      unfold sat_model_certificate_check in Hcand.
      exact Hcand.
    + exact (IH true_atoms Hfind).
Qed.

Theorem find_satisfying_assignment_sat_sound :
  forall (cnf : CNF) (candidates : list (list Atom))
         (true_atoms : list Atom),
    find_satisfying_assignment cnf candidates = Some true_atoms ->
    exists rho : Assignment,
      cnf_evalb rho cnf = true.
Proof.
  intros cnf candidates true_atoms Hfind.
  exists (finite_assignment true_atoms).
  exact
    (find_satisfying_assignment_sound
       cnf candidates true_atoms Hfind).
Qed.

Lemma find_satisfying_assignment_none_all_false :
  forall (cnf : CNF) (candidates : list (list Atom)),
    find_satisfying_assignment cnf candidates = None ->
    forall true_atoms : list Atom,
      In true_atoms candidates ->
      sat_model_certificate_check cnf true_atoms = false.
Proof.
  intros cnf candidates.
  induction candidates as [| cand tl IH]; intros Hnone true_atoms Hin.
  - simpl in Hin. contradiction.
  - simpl in Hnone.
    destruct (sat_model_certificate_check cnf cand) eqn:Hcand.
    + discriminate.
    + simpl in Hin.
      destruct Hin as [Heq | HinTl].
      * subst true_atoms. exact Hcand.
      * exact (IH Hnone true_atoms HinTl).
Qed.

Definition candidates_complete_for_cnf
           (cnf : CNF) (candidates : list (list Atom)) : Prop :=
  forall rho : Assignment,
    cnf_evalb rho cnf = true ->
    exists true_atoms : list Atom,
      In true_atoms candidates /\
      cnf_evalb (finite_assignment true_atoms) cnf = true.

Theorem find_satisfying_assignment_none_unsat_sound :
  forall (cnf : CNF) (candidates : list (list Atom)),
    candidates_complete_for_cnf cnf candidates ->
    find_satisfying_assignment cnf candidates = None ->
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf candidates Hcomplete Hnone.
  intros [rho Hsat].
  unfold candidates_complete_for_cnf in Hcomplete.
  destruct (Hcomplete rho Hsat) as [true_atoms [Hin Hmodel]].
  pose proof
    (find_satisfying_assignment_none_all_false
       cnf candidates Hnone true_atoms Hin)
    as Hfalse.
  unfold sat_model_certificate_check in Hfalse.
  rewrite Hmodel in Hfalse.
  discriminate.
Qed.

Definition option_is_none {A : Type} (o : option A) : bool :=
  match o with
  | None => true
  | Some _ => false
  end.

Lemma option_is_none_true :
  forall (A : Type) (o : option A),
    option_is_none o = true ->
    o = None.
Proof.
  intros A o H.
  destruct o as [x |].
  - simpl in H. discriminate.
  - reflexivity.
Qed.

Definition exhaustive_horn_unsat_check
           (cnf : CNF) (candidates : list (list Atom)) : bool :=
  andb
    (all_horn_clauses cnf)
    (option_is_none (find_satisfying_assignment cnf candidates)).

Definition exhaustive_antihorn_unsat_check
           (cnf : CNF) (candidates : list (list Atom)) : bool :=
  andb
    (all_antihorn_clauses cnf)
    (option_is_none (find_satisfying_assignment cnf candidates)).

Theorem exhaustive_horn_unsat_check_sound :
  forall (cnf : CNF) (candidates : list (list Atom)),
    candidates_complete_for_cnf cnf candidates ->
    exhaustive_horn_unsat_check cnf candidates = true ->
    all_horn_clauses cnf = true /\
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf candidates Hcomplete Hcheck.
  unfold exhaustive_horn_unsat_check in Hcheck.
  apply andb_true_iff in Hcheck.
  destruct Hcheck as [Hhorn HnoneBool].
  split.
  - exact Hhorn.
  - apply find_satisfying_assignment_none_unsat_sound
      with (candidates := candidates).
    + exact Hcomplete.
    + apply option_is_none_true.
      exact HnoneBool.
Qed.

Theorem exhaustive_antihorn_unsat_check_sound :
  forall (cnf : CNF) (candidates : list (list Atom)),
    candidates_complete_for_cnf cnf candidates ->
    exhaustive_antihorn_unsat_check cnf candidates = true ->
    all_antihorn_clauses cnf = true /\
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf candidates Hcomplete Hcheck.
  unfold exhaustive_antihorn_unsat_check in Hcheck.
  apply andb_true_iff in Hcheck.
  destruct Hcheck as [Hantihorn HnoneBool].
  split.
  - exact Hantihorn.
  - apply find_satisfying_assignment_none_unsat_sound
      with (candidates := candidates).
    + exact Hcomplete.
    + apply option_is_none_true.
      exact HnoneBool.
Qed.

Theorem exhaustive_horn_sat_result_sound :
  forall (cnf : CNF) (candidates : list (list Atom))
         (true_atoms : list Atom),
    all_horn_clauses cnf = true ->
    find_satisfying_assignment cnf candidates = Some true_atoms ->
    all_horn_clauses cnf = true /\
    exists rho : Assignment,
      cnf_evalb rho cnf = true.
Proof.
  intros cnf candidates true_atoms Hhorn Hfind.
  split.
  - exact Hhorn.
  - exact
      (find_satisfying_assignment_sat_sound
         cnf candidates true_atoms Hfind).
Qed.

Theorem exhaustive_antihorn_sat_result_sound :
  forall (cnf : CNF) (candidates : list (list Atom))
         (true_atoms : list Atom),
    all_antihorn_clauses cnf = true ->
    find_satisfying_assignment cnf candidates = Some true_atoms ->
    all_antihorn_clauses cnf = true /\
    exists rho : Assignment,
      cnf_evalb rho cnf = true.
Proof.
  intros cnf candidates true_atoms Hantihorn Hfind.
  split.
  - exact Hantihorn.
  - exact
      (find_satisfying_assignment_sat_sound
         cnf candidates true_atoms Hfind).
Qed.

(* ================================================================ *)
(* 17. CNF atom-support and candidate-completeness bridge             *)
(* ================================================================ *)

(* This section proves the key bridge needed before automatic powerset
   generation:

      CNF evaluation depends only on the truth values of atoms that actually
      occur in the CNF.

   Therefore, to prove that a finite candidate list is complete for a CNF, it
   is enough to show that every assignment can be represented on atoms_of_cnf
   by one candidate model.

   This avoids trusting irrelevant atoms outside the formula.
*)

Definition atom_of_signed (s : SignedAtom) : Atom :=
  s.(sa_atom).

Fixpoint atoms_of_clause (c : Clause) : list Atom :=
  match c with
  | [] => []
  | s :: tl => atom_of_signed s :: atoms_of_clause tl
  end.

Fixpoint atoms_of_cnf (cnf : CNF) : list Atom :=
  match cnf with
  | [] => []
  | c :: tl => atoms_of_clause c ++ atoms_of_cnf tl
  end.

Definition assignments_agree_on_atoms
           (atoms : list Atom)
           (rho sigma : Assignment) : Prop :=
  forall a : Atom,
    In a atoms ->
    rho a = sigma a.

Lemma signed_evalb_agree_on_atoms :
  forall (atoms : list Atom) (rho sigma : Assignment)
         (s : SignedAtom),
    In (atom_of_signed s) atoms ->
    assignments_agree_on_atoms atoms rho sigma ->
    signed_evalb rho s = signed_evalb sigma s.
Proof.
  intros atoms rho sigma s Hin Hagree.
  destruct s as [a pos].
  unfold atom_of_signed in Hin.
  unfold signed_evalb.
  simpl in *.
  destruct pos.
  - apply Hagree.
    exact Hin.
  - rewrite (Hagree a Hin).
    reflexivity.
Qed.

Lemma clause_evalb_agree_on_atoms :
  forall (c : Clause) (rho sigma : Assignment),
    assignments_agree_on_atoms (atoms_of_clause c) rho sigma ->
    clause_evalb rho c = clause_evalb sigma c.
Proof.
  intros c.
  induction c as [| s tl IH]; intros rho sigma Hagree.
  - simpl. reflexivity.
  - unfold clause_evalb.
    simpl.
    assert (Hs : signed_evalb rho s = signed_evalb sigma s).
    {
      apply signed_evalb_agree_on_atoms
        with (atoms := atom_of_signed s :: atoms_of_clause tl).
      - simpl. left. reflexivity.
      - exact Hagree.
    }
    assert
      (Htl :
         existsb (fun s0 : SignedAtom => signed_evalb rho s0) tl =
         existsb (fun s0 : SignedAtom => signed_evalb sigma s0) tl).
    {
      change (clause_evalb rho tl = clause_evalb sigma tl).
      apply IH.
      unfold assignments_agree_on_atoms in *.
      intros a Ha.
      apply Hagree.
      simpl.
      right.
      exact Ha.
    }
    rewrite Hs.
    rewrite Htl.
    reflexivity.
Qed.

Lemma cnf_evalb_agree_on_atoms :
  forall (cnf : CNF) (rho sigma : Assignment),
    assignments_agree_on_atoms (atoms_of_cnf cnf) rho sigma ->
    cnf_evalb rho cnf = cnf_evalb sigma cnf.
Proof.
  intros cnf.
  induction cnf as [| c tl IH]; intros rho sigma Hagree.
  - simpl. reflexivity.
  - unfold cnf_evalb.
    simpl.
    assert (Hc : clause_evalb rho c = clause_evalb sigma c).
    {
      apply clause_evalb_agree_on_atoms.
      unfold assignments_agree_on_atoms in *.
      intros a Ha.
      apply Hagree.
      simpl.
      apply in_or_app.
      left.
      exact Ha.
    }
    assert
      (Htl :
         forallb (fun c0 : Clause => clause_evalb rho c0) tl =
         forallb (fun c0 : Clause => clause_evalb sigma c0) tl).
    {
      change (cnf_evalb rho tl = cnf_evalb sigma tl).
      apply IH.
      unfold assignments_agree_on_atoms in *.
      intros a Ha.
      apply Hagree.
      simpl.
      apply in_or_app.
      right.
      exact Ha.
    }
    rewrite Hc.
    rewrite Htl.
    reflexivity.
Qed.
Definition candidate_represents_assignment_on_atoms
           (atoms : list Atom)
           (rho : Assignment)
           (true_atoms : list Atom) : Prop :=
  assignments_agree_on_atoms atoms rho (finite_assignment true_atoms).

Definition candidates_cover_assignments_on_atoms
           (atoms : list Atom)
           (candidates : list (list Atom)) : Prop :=
  forall rho : Assignment,
    exists true_atoms : list Atom,
      In true_atoms candidates /\
      candidate_represents_assignment_on_atoms atoms rho true_atoms.

Theorem candidates_cover_cnf_atoms_implies_complete_for_cnf :
  forall (cnf : CNF) (candidates : list (list Atom)),
    candidates_cover_assignments_on_atoms (atoms_of_cnf cnf) candidates ->
    candidates_complete_for_cnf cnf candidates.
Proof.
  intros cnf candidates Hcover.
  unfold candidates_complete_for_cnf.
  intros rho Hsat.
  unfold candidates_cover_assignments_on_atoms in Hcover.
  destruct (Hcover rho) as [true_atoms [Hin Hrep]].
  exists true_atoms.
  split.
  - exact Hin.
  - unfold candidate_represents_assignment_on_atoms in Hrep.
    pose proof
      (cnf_evalb_agree_on_atoms
         cnf rho (finite_assignment true_atoms) Hrep)
      as Heq.
    rewrite <- Heq.
    exact Hsat.
Qed.

Theorem exhaustive_solver_complete_from_cnf_atom_coverage :
  forall (cnf : CNF) (candidates : list (list Atom)),
    candidates_cover_assignments_on_atoms (atoms_of_cnf cnf) candidates ->
    find_satisfying_assignment cnf candidates = None ->
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf candidates Hcover Hnone.
  apply find_satisfying_assignment_none_unsat_sound
    with (candidates := candidates).
  - apply candidates_cover_cnf_atoms_implies_complete_for_cnf.
    exact Hcover.
  - exact Hnone.
Qed.

Theorem exhaustive_horn_unsat_from_cnf_atom_coverage :
  forall (cnf : CNF) (candidates : list (list Atom)),
    candidates_cover_assignments_on_atoms (atoms_of_cnf cnf) candidates ->
    exhaustive_horn_unsat_check cnf candidates = true ->
    all_horn_clauses cnf = true /\
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf candidates Hcover Hcheck.
  exact
    (exhaustive_horn_unsat_check_sound
       cnf
       candidates
       (candidates_cover_cnf_atoms_implies_complete_for_cnf
          cnf
          candidates
          Hcover)
       Hcheck).
Qed.

Theorem exhaustive_antihorn_unsat_from_cnf_atom_coverage :
  forall (cnf : CNF) (candidates : list (list Atom)),
    candidates_cover_assignments_on_atoms (atoms_of_cnf cnf) candidates ->
    exhaustive_antihorn_unsat_check cnf candidates = true ->
    all_antihorn_clauses cnf = true /\
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf candidates Hcover Hcheck.
  exact
    (exhaustive_antihorn_unsat_check_sound
       cnf
       candidates
       (candidates_cover_cnf_atoms_implies_complete_for_cnf
          cnf
          candidates
          Hcover)
       Hcheck).
Qed.

(* ================================================================ *)
(* 18. Powerset candidate generator and complete exhaustive solver     *)
(* ================================================================ *)

(* This section builds the automatic finite candidate generator.

   Important precision:
   Section 15 used [atom_eqb], based on [Qeq_bool], for certificate
   membership.  That equality is semantic rational equality.

   For an automatically complete exhaustive solver, we use a structural atom
   equality [atom_struct_eqb].  This lets Coq prove that every Boolean
   assignment over the concrete atoms appearing in a CNF is represented by
   one generated candidate.

   This gives a fully verified finite exhaustive solver for the concrete CNF
   syntax.  A later refinement layer can relate normalized threshold atoms to
   the semantic [Qeq_bool]-based certificate checker.
*)

Definition q_struct_eqb (x y : Q) : bool :=
  match x, y with
  | Qmake nx dx, Qmake ny dy =>
      andb (Z.eqb nx ny) (Pos.eqb dx dy)
  end.

Definition atom_struct_eqb (a b : Atom) : bool :=
  match a, b with
  | (f,t), (g,u) =>
      andb (Nat.eqb f g) (q_struct_eqb t u)
  end.

Lemma q_struct_eqb_refl :
  forall q : Q,
    q_struct_eqb q q = true.
Proof.
  intros q.
  destruct q as [n d].
  unfold q_struct_eqb.
  rewrite Z.eqb_refl.
  rewrite Pos.eqb_refl.
  reflexivity.
Qed.

Lemma q_struct_eqb_eq :
  forall x y : Q,
    q_struct_eqb x y = true ->
    x = y.
Proof.
  intros x y H.
  destruct x as [nx dx].
  destruct y as [ny dy].
  unfold q_struct_eqb in H.
  apply andb_true_iff in H.
  destruct H as [Hn Hd].
  apply Z.eqb_eq in Hn.
  apply Pos.eqb_eq in Hd.
  subst.
  reflexivity.
Qed.

Lemma atom_struct_eqb_refl :
  forall a : Atom,
    atom_struct_eqb a a = true.
Proof.
  intros a.
  destruct a as [f t].
  unfold atom_struct_eqb.
  rewrite Nat.eqb_refl.
  rewrite q_struct_eqb_refl.
  reflexivity.
Qed.

Lemma atom_struct_eqb_eq :
  forall a b : Atom,
    atom_struct_eqb a b = true ->
    a = b.
Proof.
  intros a b H.
  destruct a as [f t].
  destruct b as [g u].
  unfold atom_struct_eqb in H.
  apply andb_true_iff in H.
  destruct H as [Hf Hq].
  apply Nat.eqb_eq in Hf.
  apply q_struct_eqb_eq in Hq.
  subst.
  reflexivity.
Qed.

Fixpoint atom_struct_in_bool (a : Atom) (atoms : list Atom) : bool :=
  match atoms with
  | [] => false
  | b :: tl =>
      if atom_struct_eqb a b
      then true
      else atom_struct_in_bool a tl
  end.

Definition finite_struct_assignment (true_atoms : list Atom) : Assignment :=
  fun a => atom_struct_in_bool a true_atoms.

Fixpoint select_true_atoms_struct
         (rho : Assignment) (atoms : list Atom) : list Atom :=
  match atoms with
  | [] => []
  | a :: tl =>
      if rho a
      then a :: select_true_atoms_struct rho tl
      else select_true_atoms_struct rho tl
  end.

Fixpoint powerset_atom_candidates (atoms : list Atom) : list (list Atom) :=
  match atoms with
  | [] => [[]]
  | a :: tl =>
      let rest := powerset_atom_candidates tl in
      rest ++ map (fun candidate => a :: candidate) rest
  end.

Lemma selected_true_atoms_in_powerset :
  forall (atoms : list Atom) (rho : Assignment),
    In (select_true_atoms_struct rho atoms)
       (powerset_atom_candidates atoms).
Proof.
  intros atoms.
  induction atoms as [| a tl IH]; intros rho.
  - simpl. left. reflexivity.
  - simpl.
    destruct (rho a) eqn:Ha.
    + apply in_or_app.
      right.
      apply in_map.
      exact (IH rho).
    + apply in_or_app.
      left.
      exact (IH rho).
Qed.

Lemma atom_struct_in_bool_true_exists :
  forall (a : Atom) (atoms : list Atom),
    atom_struct_in_bool a atoms = true ->
    exists b : Atom,
      In b atoms /\
      atom_struct_eqb a b = true.
Proof.
  intros a atoms.
  induction atoms as [| b tl IH]; intros H.
  - simpl in H. discriminate.
  - simpl in H.
    destruct (atom_struct_eqb a b) eqn:Hab.
    + exists b.
      split.
      * simpl. left. reflexivity.
      * exact Hab.
    + destruct (IH H) as [c [Hin Hac]].
      exists c.
      split.
      * simpl. right. exact Hin.
      * exact Hac.
Qed.

Lemma select_true_atoms_struct_sound :
  forall (atoms : list Atom) (rho : Assignment) (a : Atom),
    In a (select_true_atoms_struct rho atoms) ->
    In a atoms /\ rho a = true.
Proof.
  intros atoms.
  induction atoms as [| b tl IH]; intros rho a Hin.
  - simpl in Hin. contradiction.
  - simpl in Hin.
    destruct (rho b) eqn:Hb.
    + simpl in Hin.
      destruct Hin as [Ha | HinTl].
      * subst a.
        split.
        -- simpl. left. reflexivity.
        -- exact Hb.
      * destruct (IH rho a HinTl) as [HinAtoms Hrho].
        split.
        -- simpl. right. exact HinAtoms.
        -- exact Hrho.
    + destruct (IH rho a Hin) as [HinAtoms Hrho].
      split.
      * simpl. right. exact HinAtoms.
      * exact Hrho.
Qed.

Lemma finite_struct_assignment_selected_false :
  forall (atoms : list Atom) (rho : Assignment) (a : Atom),
    rho a = false ->
    finite_struct_assignment
      (select_true_atoms_struct rho atoms) a = false.
Proof.
  intros atoms rho a Hrho.
  unfold finite_struct_assignment.
  destruct
    (atom_struct_in_bool a (select_true_atoms_struct rho atoms))
    eqn:HinBool.
  - exfalso.
    apply atom_struct_in_bool_true_exists in HinBool.
    destruct HinBool as [b [HinSelected Hab]].
    apply select_true_atoms_struct_sound in HinSelected.
    destruct HinSelected as [_ HrhoB].
    apply atom_struct_eqb_eq in Hab.
    subst b.
    rewrite Hrho in HrhoB.
    discriminate.
  - reflexivity.
Qed.

Lemma finite_struct_assignment_selected_agrees_on_atoms :
  forall (atoms : list Atom) (rho : Assignment) (a : Atom),
    In a atoms ->
    finite_struct_assignment
      (select_true_atoms_struct rho atoms) a = rho a.
Proof.
  intros atoms.
  induction atoms as [| b tl IH]; intros rho a Hin.
  - simpl in Hin. contradiction.
  - simpl in Hin.
    destruct Hin as [Ha | HinTl].
    + subst b.
      simpl.
      destruct (rho a) eqn:Hrho.
      * unfold finite_struct_assignment.
        simpl.
        rewrite atom_struct_eqb_refl.
        reflexivity.
      * change
          (finite_struct_assignment
             (select_true_atoms_struct rho tl) a = false).
        apply finite_struct_assignment_selected_false.
        exact Hrho.
    + simpl.
      destruct (rho b) eqn:Hb.
      * unfold finite_struct_assignment.
        simpl.
        destruct (atom_struct_eqb a b) eqn:Hab.
        -- apply atom_struct_eqb_eq in Hab.
           subst b.
           rewrite Hb.
           reflexivity.
        -- change
             (finite_struct_assignment
                (select_true_atoms_struct rho tl) a = rho a).
           apply IH.
           exact HinTl.
      * change
          (finite_struct_assignment
             (select_true_atoms_struct rho tl) a = rho a).
        apply IH.
        exact HinTl.
Qed.

Definition struct_candidates_cover_assignments_on_atoms
           (atoms : list Atom)
           (candidates : list (list Atom)) : Prop :=
  forall rho : Assignment,
    exists true_atoms : list Atom,
      In true_atoms candidates /\
      assignments_agree_on_atoms
        atoms rho (finite_struct_assignment true_atoms).

Theorem powerset_struct_candidates_cover_assignments_on_atoms :
  forall atoms : list Atom,
    struct_candidates_cover_assignments_on_atoms
      atoms
      (powerset_atom_candidates atoms).
Proof.
  intros atoms.
  unfold struct_candidates_cover_assignments_on_atoms.
  intros rho.
  exists (select_true_atoms_struct rho atoms).
  split.
  - apply selected_true_atoms_in_powerset.
  - unfold assignments_agree_on_atoms.
    intros a Hin.
    symmetry.
    apply finite_struct_assignment_selected_agrees_on_atoms.
    exact Hin.
Qed.

Definition candidates_complete_for_cnf_struct
           (cnf : CNF) (candidates : list (list Atom)) : Prop :=
  forall rho : Assignment,
    cnf_evalb rho cnf = true ->
    exists true_atoms : list Atom,
      In true_atoms candidates /\
      cnf_evalb (finite_struct_assignment true_atoms) cnf = true.

Theorem struct_candidates_cover_cnf_atoms_implies_complete_for_cnf :
  forall (cnf : CNF) (candidates : list (list Atom)),
    struct_candidates_cover_assignments_on_atoms
      (atoms_of_cnf cnf) candidates ->
    candidates_complete_for_cnf_struct cnf candidates.
Proof.
  intros cnf candidates Hcover.
  unfold candidates_complete_for_cnf_struct.
  intros rho Hsat.
  unfold struct_candidates_cover_assignments_on_atoms in Hcover.
  destruct (Hcover rho) as [true_atoms [Hin Hagree]].
  exists true_atoms.
  split.
  - exact Hin.
  - pose proof
      (cnf_evalb_agree_on_atoms
         cnf rho (finite_struct_assignment true_atoms) Hagree)
      as Heq.
    rewrite <- Heq.
    exact Hsat.
Qed.

Theorem powerset_struct_candidates_complete_for_cnf :
  forall cnf : CNF,
    candidates_complete_for_cnf_struct
      cnf
      (powerset_atom_candidates (atoms_of_cnf cnf)).
Proof.
  intros cnf.
  apply struct_candidates_cover_cnf_atoms_implies_complete_for_cnf.
  apply powerset_struct_candidates_cover_assignments_on_atoms.
Qed.

Definition struct_sat_model_check
           (cnf : CNF) (true_atoms : list Atom) : bool :=
  cnf_evalb (finite_struct_assignment true_atoms) cnf.

Fixpoint find_satisfying_struct_assignment
         (cnf : CNF) (candidates : list (list Atom))
  : option (list Atom) :=
  match candidates with
  | [] => None
  | true_atoms :: tl =>
      if struct_sat_model_check cnf true_atoms
      then Some true_atoms
      else find_satisfying_struct_assignment cnf tl
  end.

Theorem find_satisfying_struct_assignment_sound :
  forall (cnf : CNF) (candidates : list (list Atom))
         (true_atoms : list Atom),
    find_satisfying_struct_assignment cnf candidates = Some true_atoms ->
    exists rho : Assignment,
      cnf_evalb rho cnf = true.
Proof.
  intros cnf candidates.
  induction candidates as [| cand tl IH]; intros true_atoms Hfind.
  - simpl in Hfind. discriminate.
  - simpl in Hfind.
    destruct (struct_sat_model_check cnf cand) eqn:Hcand.
    + injection Hfind as Hmodel.
      subst true_atoms.
      exists (finite_struct_assignment cand).
      unfold struct_sat_model_check in Hcand.
      exact Hcand.
    + exact (IH true_atoms Hfind).
Qed.

Lemma find_satisfying_struct_assignment_none_all_false :
  forall (cnf : CNF) (candidates : list (list Atom)),
    find_satisfying_struct_assignment cnf candidates = None ->
    forall true_atoms : list Atom,
      In true_atoms candidates ->
      struct_sat_model_check cnf true_atoms = false.
Proof.
  intros cnf candidates.
  induction candidates as [| cand tl IH]; intros Hnone true_atoms Hin.
  - simpl in Hin. contradiction.
  - simpl in Hnone.
    destruct (struct_sat_model_check cnf cand) eqn:Hcand.
    + discriminate.
    + simpl in Hin.
      destruct Hin as [Heq | HinTl].
      * subst true_atoms. exact Hcand.
      * exact (IH Hnone true_atoms HinTl).
Qed.

Theorem find_satisfying_struct_assignment_none_unsat_sound :
  forall (cnf : CNF) (candidates : list (list Atom)),
    candidates_complete_for_cnf_struct cnf candidates ->
    find_satisfying_struct_assignment cnf candidates = None ->
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf candidates Hcomplete Hnone.
  intros [rho Hsat].
  unfold candidates_complete_for_cnf_struct in Hcomplete.
  destruct (Hcomplete rho Hsat) as [true_atoms [Hin Hmodel]].
  pose proof
    (find_satisfying_struct_assignment_none_all_false
       cnf candidates Hnone true_atoms Hin)
    as Hfalse.
  unfold struct_sat_model_check in Hfalse.
  rewrite Hmodel in Hfalse.
  discriminate.
Qed.

Theorem powerset_struct_exhaustive_solver_unsat_sound :
  forall cnf : CNF,
    find_satisfying_struct_assignment
      cnf
      (powerset_atom_candidates (atoms_of_cnf cnf)) = None ->
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf Hnone.
  apply find_satisfying_struct_assignment_none_unsat_sound
    with (candidates := powerset_atom_candidates (atoms_of_cnf cnf)).
  - apply powerset_struct_candidates_complete_for_cnf.
  - exact Hnone.
Qed.

Theorem powerset_struct_exhaustive_solver_sat_sound :
  forall (cnf : CNF) (true_atoms : list Atom),
    find_satisfying_struct_assignment
      cnf
      (powerset_atom_candidates (atoms_of_cnf cnf)) = Some true_atoms ->
    exists rho : Assignment,
      cnf_evalb rho cnf = true.
Proof.
  intros cnf true_atoms Hfind.
  exact
    (find_satisfying_struct_assignment_sound
       cnf
       (powerset_atom_candidates (atoms_of_cnf cnf))
       true_atoms
       Hfind).
Qed.

Definition powerset_horn_exhaustive_unsat_check (cnf : CNF) : bool :=
  andb
    (all_horn_clauses cnf)
    (option_is_none
       (find_satisfying_struct_assignment
          cnf
          (powerset_atom_candidates (atoms_of_cnf cnf)))).

Definition powerset_antihorn_exhaustive_unsat_check (cnf : CNF) : bool :=
  andb
    (all_antihorn_clauses cnf)
    (option_is_none
       (find_satisfying_struct_assignment
          cnf
          (powerset_atom_candidates (atoms_of_cnf cnf)))).

Theorem powerset_horn_exhaustive_unsat_check_sound :
  forall cnf : CNF,
    powerset_horn_exhaustive_unsat_check cnf = true ->
    all_horn_clauses cnf = true /\
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf Hcheck.
  unfold powerset_horn_exhaustive_unsat_check in Hcheck.
  apply andb_true_iff in Hcheck.
  destruct Hcheck as [Hhorn HnoneBool].
  split.
  - exact Hhorn.
  - apply powerset_struct_exhaustive_solver_unsat_sound.
    apply option_is_none_true.
    exact HnoneBool.
Qed.

Theorem powerset_antihorn_exhaustive_unsat_check_sound :
  forall cnf : CNF,
    powerset_antihorn_exhaustive_unsat_check cnf = true ->
    all_antihorn_clauses cnf = true /\
    ~ exists rho : Assignment,
        cnf_evalb rho cnf = true.
Proof.
  intros cnf Hcheck.
  unfold powerset_antihorn_exhaustive_unsat_check in Hcheck.
  apply andb_true_iff in Hcheck.
  destruct Hcheck as [Hantihorn HnoneBool].
  split.
  - exact Hantihorn.
  - apply powerset_struct_exhaustive_solver_unsat_sound.
    apply option_is_none_true.
    exact HnoneBool.
Qed.

(* ================================================================ *)
(* 19. Selected-feature exhaustive path-blocking checker              *)
(* ================================================================ *)

(* This section connects the complete exhaustive CNF solver to the weak-AXp
   path-blocking condition.

   For a selected feature set S and reference valuation x0, we build unit
   clauses forcing every selected threshold atom on the path to agree with x0.
   Then we exhaustively check UNSAT of:

        selected-unit-clauses(x0,S,path) ++ path_clauses(path)

   If this selected CNF is UNSAT, then the corresponding Boolean selected-path
   satisfiability predicate is impossible.

   This is a sound checker for weak-AXp opposite-path blocking.  It is not yet
   a completeness/reflection theorem because order-consistency is still a Prop
   side-condition, not encoded as CNF here.
*)

Definition unit_clause_for_selected_atom
           (x0 : Valuation) (a : Atom) : Clause :=
  [mkSignedAtom a (induced_assignment x0 a)].

Fixpoint selection_clauses_from_atoms
         (x0 : Valuation) (S : list Feature) (atoms : list Atom)
  : CNF :=
  match atoms with
  | [] => []
  | (f,t) :: tl =>
      if feature_in_bool f S
      then unit_clause_for_selected_atom x0 (f,t)
             :: selection_clauses_from_atoms x0 S tl
      else selection_clauses_from_atoms x0 S tl
  end.

Definition selected_path_cnf
           (x0 : Valuation) (S : list Feature) (path : Path) : CNF :=
  selection_clauses_from_atoms x0 S (atoms_of_path path)
  ++ path_clauses path.

Lemma unit_clause_for_selected_atom_complete :
  forall (x0 : Valuation) (rho : Assignment) (a : Atom),
    induced_assignment x0 a = rho a ->
    clause_evalb rho (unit_clause_for_selected_atom x0 a) = true.
Proof.
  intros x0 rho a Hagree.
  unfold unit_clause_for_selected_atom.
  unfold clause_evalb.
  simpl.
  unfold signed_evalb.
  simpl.
  rewrite Hagree.
  destruct (rho a); reflexivity.
Qed.

Lemma selection_clauses_from_atoms_complete :
  forall (x0 : Valuation) (S : list Feature)
         (atoms : list Atom) (rho : Assignment),
    selected_atom_agreement x0 S atoms rho ->
    cnf_evalb rho (selection_clauses_from_atoms x0 S atoms) = true.
Proof.
  intros x0 S atoms.
  induction atoms as [| a tl IH]; intros rho Hagree.
  - simpl. reflexivity.
  - destruct a as [f t].
    simpl.
    destruct (feature_in_bool f S) eqn:Hsel.
    + unfold cnf_evalb.
      simpl.
      apply andb_true_iff.
      split.
      * change
          (clause_evalb rho
             (unit_clause_for_selected_atom x0 (f, t)) = true).
        apply unit_clause_for_selected_atom_complete.
        apply Hagree.
        -- simpl. left. reflexivity.
        -- apply feature_in_bool_true_in.
           exact Hsel.
      * change
          (cnf_evalb rho
             (selection_clauses_from_atoms x0 S tl) = true).
        apply IH.
        unfold selected_atom_agreement in *.
        intros f' t' Hin Hf'.
        apply Hagree.
        -- simpl. right. exact Hin.
        -- exact Hf'.
    + apply IH.
      unfold selected_atom_agreement in *.
      intros f' t' Hin Hf'.
      apply Hagree.
      * simpl. right. exact Hin.
      * exact Hf'.
Qed.

Theorem selected_path_cnf_complete_for_boolean_selected_path :
  forall (x0 : Valuation) (S : list Feature)
         (path : Path) (rho : Assignment),
    selected_atom_agreement x0 S (atoms_of_path path) rho ->
    cnf_evalb rho (path_clauses path) = true ->
    cnf_evalb rho (selected_path_cnf x0 S path) = true.
Proof.
  intros x0 S path rho Hagree Hpath.
  unfold selected_path_cnf.
  unfold cnf_evalb.
  rewrite forallb_app.
  apply andb_true_iff.
  split.
  - change
      (cnf_evalb rho
         (selection_clauses_from_atoms x0 S (atoms_of_path path)) = true).
    apply selection_clauses_from_atoms_complete.
    exact Hagree.
  - exact Hpath.
Qed.

Definition selected_path_exhaustive_unsat_check
           (x0 : Valuation) (S : list Feature) (path : Path) : bool :=
  option_is_none
    (find_satisfying_struct_assignment
       (selected_path_cnf x0 S path)
       (powerset_atom_candidates
          (atoms_of_cnf (selected_path_cnf x0 S path)))).

Theorem selected_path_exhaustive_unsat_check_sound :
  forall (x0 : Valuation) (S : list Feature) (path : Path),
    selected_path_exhaustive_unsat_check x0 S path = true ->
    ~ boolean_path_satisfiable_with_order_under_selection x0 S path.
Proof.
  intros x0 S path Hcheck.
  unfold selected_path_exhaustive_unsat_check in Hcheck.
  pose proof
    (option_is_none_true
       (list Atom)
       (find_satisfying_struct_assignment
          (selected_path_cnf x0 S path)
          (powerset_atom_candidates
             (atoms_of_cnf (selected_path_cnf x0 S path))))
       Hcheck)
    as Hnone.
  pose proof
    (powerset_struct_exhaustive_solver_unsat_sound
       (selected_path_cnf x0 S path)
       Hnone)
    as Hunsat.
  intros HsatSelected.
  unfold boolean_path_satisfiable_with_order_under_selection
    in HsatSelected.
  destruct HsatSelected as [rho [Horder [Hagree Hpath]]].
  apply Hunsat.
  exists rho.
  apply selected_path_cnf_complete_for_boolean_selected_path.
  - exact Hagree.
  - exact Hpath.
Qed.

Definition all_selected_paths_exhaustive_unsat_check
           (x0 : Valuation) (S : list Feature)
           (paths : list Path) : bool :=
  forallb (selected_path_exhaustive_unsat_check x0 S) paths.

Theorem all_selected_paths_exhaustive_unsat_check_sound :
  forall (x0 : Valuation) (S : list Feature)
         (opposite_paths : list Path),
    all_selected_paths_exhaustive_unsat_check
      x0 S opposite_paths = true ->
    weak_axp_selected_path_condition_boolean x0 S opposite_paths.
Proof.
  intros x0 S opposite_paths Hcheck.
  unfold weak_axp_selected_path_condition_boolean.
  unfold boolean_opposite_paths_blocked_under_selection.
  intros path Hin.
  apply selected_path_exhaustive_unsat_check_sound.
  unfold all_selected_paths_exhaustive_unsat_check in Hcheck.
  apply forallb_forall with (x := path) in Hcheck.
  - exact Hcheck.
  - exact Hin.
Qed.

Definition verified_exhaustive_weak_axp_checker
           (x0 : Valuation) (opposite_paths : list Path)
           (S : list Feature) : bool :=
  all_selected_paths_exhaustive_unsat_check x0 S opposite_paths.

Theorem verified_exhaustive_weak_axp_checker_sound :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (S : list Feature),
    verified_exhaustive_weak_axp_checker x0 opposite_paths S = true ->
    weak_axp_selected_path_condition_boolean x0 S opposite_paths.
Proof.
  intros x0 opposite_paths S Hcheck.
  unfold verified_exhaustive_weak_axp_checker in Hcheck.
  apply all_selected_paths_exhaustive_unsat_check_sound.
  exact Hcheck.
Qed.

Theorem verified_exhaustive_weak_axp_checker_numeric_sound :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (S : list Feature),
    verified_exhaustive_weak_axp_checker x0 opposite_paths S = true ->
    weak_axp_selected_path_condition_numeric x0 S opposite_paths.
Proof.
  intros x0 opposite_paths S Hcheck.
  destruct
    (weak_axp_selected_path_condition_equivalence
       x0 S opposite_paths)
    as [_ Hbool_to_num].
  apply Hbool_to_num.
  apply verified_exhaustive_weak_axp_checker_sound.
  exact Hcheck.
Qed.

(* ================================================================ *)
(* 21. Structural order CNF: soundness and completeness               *)
(* ================================================================ *)

(* Full replacement for Section 21.

   Delete the old Section 21 completely, from the line:

      (* 21. Structural order CNF: soundness and completeness *)

   until the end of that section, then paste this block.

   Key fix:
   the old generator compared each atom only with the suffix [a :: tl].
   That is incomplete.  This replacement compares every source atom against
   the full atom support.
*)

Fixpoint order_clauses_for_atom
         (a : Atom) (atoms : list Atom) {struct atoms} : CNF :=
  match a with
  | (f,t1) =>
      match atoms with
      | [] => []
      | (g,t2) :: tl =>
          if andb (Nat.eqb f g) (Qle_bool t1 t2)
          then structural_order_clause f t1 t2
                 :: order_clauses_for_atom (f,t1) tl
          else order_clauses_for_atom (f,t1) tl
      end
  end.

Fixpoint structural_order_clauses_from_atoms_aux
         (all_atoms todo : list Atom) {struct todo} : CNF :=
  match todo with
  | [] => []
  | a :: tl =>
      order_clauses_for_atom a all_atoms
      ++ structural_order_clauses_from_atoms_aux all_atoms tl
  end.

Definition structural_order_clauses_from_atoms
           (atoms : list Atom) : CNF :=
  structural_order_clauses_from_atoms_aux atoms atoms.

Lemma structural_order_clause_enforces_order :
  forall (rho : Assignment) (f : Feature) (t1 t2 : Threshold),
    clause_evalb rho (structural_order_clause f t1 t2) = true ->
    rho (f,t2) = true ->
    rho (f,t1) = true.
Proof.
  intros rho f t1 t2 Hclause Htrue2.
  unfold structural_order_clause in Hclause.
  unfold clause_evalb in Hclause.
  simpl in Hclause.
  unfold signed_evalb in Hclause.
  simpl in Hclause.
  rewrite Htrue2 in Hclause.
  simpl in Hclause.
  rewrite orb_false_r in Hclause.
  exact Hclause.
Qed.

Lemma structural_order_clause_satisfied_by_order_consistent :
  forall (atoms : list Atom) (rho : Assignment)
         (f : Feature) (t1 t2 : Threshold),
    order_consistent_on atoms rho ->
    In (f,t1) atoms ->
    In (f,t2) atoms ->
    (t1 <= t2)%Q ->
    clause_evalb rho (structural_order_clause f t1 t2) = true.
Proof.
  intros atoms rho f t1 t2 Hord Hin1 Hin2 Hle.
  unfold structural_order_clause.
  unfold clause_evalb.
  simpl.
  unfold signed_evalb.
  simpl.
  destruct (rho (f,t2)) eqn:Htrue2.
  - simpl.
    rewrite (Hord f t1 t2 Hin1 Hin2 Hle Htrue2).
    simpl.
    reflexivity.
  - simpl.
    reflexivity.
Qed.

Lemma order_clause_for_atom_in :
  forall (f : Feature) (t1 t2 : Threshold)
         (atoms : list Atom),
    In (f,t2) atoms ->
    (t1 <= t2)%Q ->
    In (structural_order_clause f t1 t2)
       (order_clauses_for_atom (f,t1) atoms).
Proof.
  intros f t1 t2 atoms.
  induction atoms as [| a tl IH]; intros Hin Hle.
  - simpl in Hin. contradiction.
  - destruct a as [g u].
    simpl in Hin.
    simpl.
    destruct Hin as [Hhead | Htail].
    + inversion Hhead; subst g u.
      rewrite Nat.eqb_refl.
      assert (Hleb : Qle_bool t1 t2 = true).
      {
        apply Qle_bool_iff.
        exact Hle.
      }
      rewrite Hleb.
      simpl.
      left.
      reflexivity.
    + destruct (andb (Nat.eqb f g) (Qle_bool t1 u)) eqn:Htest.
      * simpl. right.
        apply IH.
        -- exact Htail.
        -- exact Hle.
      * apply IH.
        -- exact Htail.
        -- exact Hle.
Qed.

Lemma structural_order_clause_in_generated_aux :
  forall (all_atoms todo : list Atom)
         (f : Feature) (t1 t2 : Threshold),
    In (f,t1) todo ->
    In (f,t2) all_atoms ->
    (t1 <= t2)%Q ->
    In (structural_order_clause f t1 t2)
       (structural_order_clauses_from_atoms_aux all_atoms todo).
Proof.
  intros all_atoms todo.
  induction todo as [| a tl IH];
    intros f t1 t2 Hin1 Hin2 Hle.
  - simpl in Hin1. contradiction.
  - simpl in Hin1.
    simpl.
    destruct Hin1 as [Hhead | HinTail].
    + destruct a as [g u].
      inversion Hhead; subst g u.
      apply in_or_app.
      left.
      apply order_clause_for_atom_in.
      * exact Hin2.
      * exact Hle.
    + apply in_or_app.
      right.
      apply IH.
      * exact HinTail.
      * exact Hin2.
      * exact Hle.
Qed.

Lemma structural_order_clause_in_generated :
  forall (atoms : list Atom)
         (f : Feature) (t1 t2 : Threshold),
    In (f,t1) atoms ->
    In (f,t2) atoms ->
    (t1 <= t2)%Q ->
    In (structural_order_clause f t1 t2)
       (structural_order_clauses_from_atoms atoms).
Proof.
  intros atoms f t1 t2 Hin1 Hin2 Hle.
  unfold structural_order_clauses_from_atoms.
  apply structural_order_clause_in_generated_aux.
  - exact Hin1.
  - exact Hin2.
  - exact Hle.
Qed.

Theorem structural_order_clauses_complete_for_order_consistency :
  forall (atoms : list Atom) (rho : Assignment),
    cnf_evalb rho (structural_order_clauses_from_atoms atoms) = true ->
    order_consistent_on atoms rho.
Proof.
  intros atoms rho Hcnf.
  unfold order_consistent_on.
  intros f t1 t2 Hin1 Hin2 Hle Htrue2.
  pose proof
    (structural_order_clause_in_generated
       atoms f t1 t2 Hin1 Hin2 Hle)
    as HinClause.
  unfold cnf_evalb in Hcnf.
  apply forallb_forall
    with (x := structural_order_clause f t1 t2) in Hcnf.
  - eapply structural_order_clause_enforces_order.
    + exact Hcnf.
    + exact Htrue2.
  - exact HinClause.
Qed.

Lemma order_clauses_for_atom_sound :
  forall (all_atoms atoms : list Atom)
         (rho : Assignment) (a : Atom),
    In a all_atoms ->
    (forall b : Atom, In b atoms -> In b all_atoms) ->
    order_consistent_on all_atoms rho ->
    cnf_evalb rho (order_clauses_for_atom a atoms) = true.
Proof.
  intros all_atoms atoms.
  induction atoms as [| b tl IH];
    intros rho a HinA Hsub Hord.
  - destruct a as [f t1].
    simpl.
    reflexivity.
  - destruct a as [f t1].
    destruct b as [g t2].
    simpl.
    destruct (andb (Nat.eqb f g) (Qle_bool t1 t2)) eqn:Htest.
    + unfold cnf_evalb.
      simpl.
      apply andb_true_iff.
      split.
      * apply andb_true_iff in Htest.
        destruct Htest as [Hfg HleBool].
        apply Nat.eqb_eq in Hfg.
        apply Qle_bool_iff in HleBool.
        subst g.
        apply structural_order_clause_satisfied_by_order_consistent
          with (atoms := all_atoms).
        -- exact Hord.
        -- exact HinA.
        -- apply Hsub. simpl. left. reflexivity.
        -- exact HleBool.
      * change
          (cnf_evalb rho
             (order_clauses_for_atom (f, t1) tl) = true).
        apply IH.
        -- exact HinA.
        -- intros c Hc.
           apply Hsub.
           simpl.
           right.
           exact Hc.
        -- exact Hord.
    + apply IH.
      * exact HinA.
      * intros c Hc.
        apply Hsub.
        simpl.
        right.
        exact Hc.
      * exact Hord.
Qed.

Lemma structural_order_clauses_aux_sound_from_order_consistency :
  forall (all_atoms todo : list Atom) (rho : Assignment),
    (forall a : Atom, In a todo -> In a all_atoms) ->
    order_consistent_on all_atoms rho ->
    cnf_evalb rho
      (structural_order_clauses_from_atoms_aux all_atoms todo) = true.
Proof.
  intros all_atoms todo.
  induction todo as [| a tl IH]; intros rho Hsub Hord.
  - simpl. reflexivity.
  - simpl.
    rewrite cnf_evalb_app.
    apply andb_true_iff.
    split.
    + apply order_clauses_for_atom_sound
        with (all_atoms := all_atoms).
      * apply Hsub.
        simpl. left. reflexivity.
      * intros b Hb.
        exact Hb.
      * exact Hord.
    + apply IH.
      * intros b Hb.
        apply Hsub.
        simpl. right. exact Hb.
      * exact Hord.
Qed.

Theorem structural_order_clauses_sound_from_order_consistency :
  forall (atoms : list Atom) (rho : Assignment),
    order_consistent_on atoms rho ->
    cnf_evalb rho (structural_order_clauses_from_atoms atoms) = true.
Proof.
  intros atoms rho Hord.
  unfold structural_order_clauses_from_atoms.
  apply structural_order_clauses_aux_sound_from_order_consistency.
  - intros a Ha. exact Ha.
  - exact Hord.
Qed.

Theorem structural_order_clauses_order_consistency_equiv :
  forall (atoms : list Atom) (rho : Assignment),
    cnf_evalb rho (structural_order_clauses_from_atoms atoms) = true <->
    order_consistent_on atoms rho.
Proof.
  intros atoms rho.
  split.
  - apply structural_order_clauses_complete_for_order_consistency.
  - apply structural_order_clauses_sound_from_order_consistency.
Qed.

Lemma order_clauses_for_atom_are_horn :
  forall (a : Atom) (atoms : list Atom),
    all_horn_clauses (order_clauses_for_atom a atoms) = true.
Proof.
  intros a atoms.
  induction atoms as [| b tl IH].
  - destruct a as [f t].
    simpl.
    reflexivity.
  - destruct a as [f t1].
    destruct b as [g t2].
    simpl.
    destruct (andb (Nat.eqb f g) (Qle_bool t1 t2)) eqn:Htest.
    + exact IH.
    + exact IH.
Qed.

Lemma order_clauses_for_atom_are_antihorn :
  forall (a : Atom) (atoms : list Atom),
    all_antihorn_clauses (order_clauses_for_atom a atoms) = true.
Proof.
  intros a atoms.
  induction atoms as [| b tl IH].
  - destruct a as [f t].
    simpl.
    reflexivity.
  - destruct a as [f t1].
    destruct b as [g t2].
    simpl.
    destruct (andb (Nat.eqb f g) (Qle_bool t1 t2)) eqn:Htest.
    + exact IH.
    + exact IH.
Qed.

Lemma structural_order_clauses_aux_are_horn :
  forall (all_atoms todo : list Atom),
    all_horn_clauses
      (structural_order_clauses_from_atoms_aux all_atoms todo) = true.
Proof.
  intros all_atoms todo.
  induction todo as [| a tl IH].
  - simpl. reflexivity.
  - simpl.
    rewrite all_horn_clauses_app.
    apply andb_true_iff.
    split.
    + apply order_clauses_for_atom_are_horn.
    + exact IH.
Qed.

Lemma structural_order_clauses_aux_are_antihorn :
  forall (all_atoms todo : list Atom),
    all_antihorn_clauses
      (structural_order_clauses_from_atoms_aux all_atoms todo) = true.
Proof.
  intros all_atoms todo.
  induction todo as [| a tl IH].
  - simpl. reflexivity.
  - simpl.
    rewrite all_antihorn_clauses_app.
    apply andb_true_iff.
    split.
    + apply order_clauses_for_atom_are_antihorn.
    + exact IH.
Qed.

Theorem structural_order_clauses_are_horn :
  forall atoms : list Atom,
    all_horn_clauses
      (structural_order_clauses_from_atoms atoms) = true.
Proof.
  intros atoms.
  unfold structural_order_clauses_from_atoms.
  apply structural_order_clauses_aux_are_horn.
Qed.

Theorem structural_order_clauses_are_antihorn :
  forall atoms : list Atom,
    all_antihorn_clauses
      (structural_order_clauses_from_atoms atoms) = true.
Proof.
  intros atoms.
  unfold structural_order_clauses_from_atoms.
  apply structural_order_clauses_aux_are_antihorn.
Qed.


(* ================================================================ *)
(* 22. Fully ordered selected-path CNF checker -- corrected version   *)
(* ================================================================ *)

(* Delete the old Section 22 completely and paste this block.

   This version fixes:
   - missing Horn/AntiHorn lemmas for selection clauses;
   - incorrect use of powerset_struct_exhaustive_solver_unsat_sound:
     that theorem expects equality to None, so we first use option_is_none_true.
*)

Definition ordered_selected_path_cnf
           (x0 : Valuation) (S : list Feature) (path : Path) : CNF :=
  selection_clauses_from_atoms x0 S (atoms_of_path path)
  ++ structural_order_clauses_from_atoms (atoms_of_path path)
  ++ path_clauses path.

Lemma unit_clause_for_selected_atom_sound :
  forall (x0 : Valuation) (rho : Assignment) (a : Atom),
    clause_evalb rho (unit_clause_for_selected_atom x0 a) = true ->
    induced_assignment x0 a = rho a.
Proof.
  intros x0 rho a Hclause.
  unfold unit_clause_for_selected_atom in Hclause.
  unfold clause_evalb in Hclause.
  simpl in Hclause.
  unfold signed_evalb in Hclause.
  simpl in Hclause.
  destruct (induced_assignment x0 a) eqn:Hind;
    destruct (rho a) eqn:Hrho;
    simpl in Hclause;
    try reflexivity;
    discriminate.
Qed.

Lemma selection_clauses_from_atoms_sound :
  forall (x0 : Valuation) (S : list Feature)
         (atoms : list Atom) (rho : Assignment),
    cnf_evalb rho (selection_clauses_from_atoms x0 S atoms) = true ->
    selected_atom_agreement x0 S atoms rho.
Proof.
  intros x0 S atoms.
  induction atoms as [| a tl IH]; intros rho Hcnf.
  - unfold selected_atom_agreement.
    intros f t Hin HinS.
    simpl in Hin.
    contradiction.
  - destruct a as [f t].
    simpl in Hcnf.
    destruct (feature_in_bool f S) eqn:Hsel.
    + unfold cnf_evalb in Hcnf.
      simpl in Hcnf.
      apply andb_true_iff in Hcnf.
      destruct Hcnf as [Hunit Htl].
      unfold selected_atom_agreement.
      intros f' t' Hin HinS.
      simpl in Hin.
      destruct Hin as [Hhead | HinTail].
      * inversion Hhead; subst f' t'.
        apply unit_clause_for_selected_atom_sound.
        change
          (clause_evalb rho
             (unit_clause_for_selected_atom x0 (f, t)) = true).
        exact Hunit.
      * apply IH.
        -- change
             (cnf_evalb rho
                (selection_clauses_from_atoms x0 S tl) = true).
           exact Htl.
        -- exact HinTail.
        -- exact HinS.
    + unfold selected_atom_agreement.
      intros f' t' Hin HinS.
      simpl in Hin.
      destruct Hin as [Hhead | HinTail].
      * inversion Hhead; subst f' t'.
        rewrite (feature_in_bool_in_true f S HinS) in Hsel.
        discriminate.
      * apply IH.
        -- change
             (cnf_evalb rho
                (selection_clauses_from_atoms x0 S tl) = true).
           exact Hcnf.
        -- exact HinTail.
        -- exact HinS.
Qed.


Lemma ordered_selection_clauses_from_atoms_are_horn :
  forall (x0 : Valuation) (S : list Feature) (atoms : list Atom),
    all_horn_clauses
      (selection_clauses_from_atoms x0 S atoms) = true.
Proof.
  intros x0 S atoms.
  induction atoms as [| a tl IH].
  - simpl. reflexivity.
  - destruct a as [f t].
    simpl.
    destruct (feature_in_bool f S) eqn:Hsel.
    + unfold all_horn_clauses.
      simpl.
      apply andb_true_iff.
      split.
      * unfold unit_clause_for_selected_atom.
        apply singleton_clause_is_horn.
      * change
          (all_horn_clauses
             (selection_clauses_from_atoms x0 S tl) = true).
        exact IH.
    + exact IH.
Qed.

Lemma ordered_selection_clauses_from_atoms_are_antihorn :
  forall (x0 : Valuation) (S : list Feature) (atoms : list Atom),
    all_antihorn_clauses
      (selection_clauses_from_atoms x0 S atoms) = true.
Proof.
  intros x0 S atoms.
  induction atoms as [| a tl IH].
  - simpl. reflexivity.
  - destruct a as [f t].
    simpl.
    destruct (feature_in_bool f S) eqn:Hsel.
    + unfold all_antihorn_clauses.
      simpl.
      apply andb_true_iff.
      split.
      * unfold unit_clause_for_selected_atom.
        apply singleton_clause_is_antihorn.
      * change
          (all_antihorn_clauses
             (selection_clauses_from_atoms x0 S tl) = true).
        exact IH.
    + exact IH.
Qed.

Theorem ordered_selected_path_cnf_complete_for_boolean_selected_path :
  forall (x0 : Valuation) (S : list Feature)
         (path : Path) (rho : Assignment),
    order_consistent_on (atoms_of_path path) rho ->
    selected_atom_agreement x0 S (atoms_of_path path) rho ->
    cnf_evalb rho (path_clauses path) = true ->
    cnf_evalb rho (ordered_selected_path_cnf x0 S path) = true.
Proof.
  intros x0 S path rho Hord Hagree Hpath.
  unfold ordered_selected_path_cnf.
  rewrite cnf_evalb_app.
  apply andb_true_iff.
  split.
  - apply selection_clauses_from_atoms_complete.
    exact Hagree.
  - rewrite cnf_evalb_app.
    apply andb_true_iff.
    split.
    + apply structural_order_clauses_sound_from_order_consistency.
      exact Hord.
    + exact Hpath.
Qed.

Theorem ordered_selected_path_cnf_sound_to_boolean_selected_path :
  forall (x0 : Valuation) (S : list Feature)
         (path : Path) (rho : Assignment),
    cnf_evalb rho (ordered_selected_path_cnf x0 S path) = true ->
    boolean_path_satisfiable_with_order_under_selection x0 S path.
Proof.
  intros x0 S path rho Hcnf.
  unfold ordered_selected_path_cnf in Hcnf.
  rewrite cnf_evalb_app in Hcnf.
  apply andb_true_iff in Hcnf.
  destruct Hcnf as [Hselection Hrest].
  rewrite cnf_evalb_app in Hrest.
  apply andb_true_iff in Hrest.
  destruct Hrest as [Horder Hpath].
  exists rho.
  split.
  - apply structural_order_clauses_complete_for_order_consistency.
    exact Horder.
  - split.
    + apply selection_clauses_from_atoms_sound.
      exact Hselection.
    + exact Hpath.
Qed.

Theorem ordered_selected_path_cnf_model_equiv_boolean_selected_path :
  forall (x0 : Valuation) (S : list Feature) (path : Path),
    (exists rho : Assignment,
        cnf_evalb rho (ordered_selected_path_cnf x0 S path) = true)
    <->
    boolean_path_satisfiable_with_order_under_selection x0 S path.
Proof.
  intros x0 S path.
  split.
  - intros [rho Hmodel].
    apply ordered_selected_path_cnf_sound_to_boolean_selected_path
      with (rho := rho).
    exact Hmodel.
  - intros [rho [Hord [Hagree Hpath]]].
    exists rho.
    apply ordered_selected_path_cnf_complete_for_boolean_selected_path.
    + exact Hord.
    + exact Hagree.
    + exact Hpath.
Qed.

Theorem ordered_selected_path_cnf_preserves_horn :
  forall (x0 : Valuation) (S : list Feature) (path : Path),
    all_horn_clauses (path_clauses path) = true ->
    all_horn_clauses (ordered_selected_path_cnf x0 S path) = true.
Proof.
  intros x0 S path Hpath.
  unfold ordered_selected_path_cnf.
  rewrite all_horn_clauses_app.
  apply andb_true_iff.
  split.
  - change
      (all_horn_clauses
         (selection_clauses_from_atoms x0 S (atoms_of_path path)) = true).
    apply ordered_selection_clauses_from_atoms_are_horn.
  - rewrite all_horn_clauses_app.
    apply andb_true_iff.
    split.
    + change
        (all_horn_clauses
           (structural_order_clauses_from_atoms (atoms_of_path path)) = true).
      apply structural_order_clauses_are_horn.
    + exact Hpath.
Qed.

Theorem ordered_selected_path_cnf_preserves_antihorn :
  forall (x0 : Valuation) (S : list Feature) (path : Path),
    all_antihorn_clauses (path_clauses path) = true ->
    all_antihorn_clauses (ordered_selected_path_cnf x0 S path) = true.
Proof.
  intros x0 S path Hpath.
  unfold ordered_selected_path_cnf.
  rewrite all_antihorn_clauses_app.
  apply andb_true_iff.
  split.
  - change
      (all_antihorn_clauses
         (selection_clauses_from_atoms x0 S (atoms_of_path path)) = true).
    apply ordered_selection_clauses_from_atoms_are_antihorn.
  - rewrite all_antihorn_clauses_app.
    apply andb_true_iff.
    split.
    + change
        (all_antihorn_clauses
           (structural_order_clauses_from_atoms (atoms_of_path path)) = true).
      apply structural_order_clauses_are_antihorn.
    + exact Hpath.
Qed.

Theorem ordered_selected_path_cnf_is_horn_for_horn_path :
  forall (x0 : Valuation) (S : list Feature) (path : Path),
    path_predicates_in_family FamilyHorn path ->
    all_horn_clauses (ordered_selected_path_cnf x0 S path) = true.
Proof.
  intros x0 S path Hfam.
  apply ordered_selected_path_cnf_preserves_horn.
  apply horn_path_clauses_are_horn.
  exact Hfam.
Qed.

Theorem ordered_selected_path_cnf_is_antihorn_for_antihorn_path :
  forall (x0 : Valuation) (S : list Feature) (path : Path),
    path_predicates_in_family FamilyAntiHorn path ->
    all_antihorn_clauses (ordered_selected_path_cnf x0 S path) = true.
Proof.
  intros x0 S path Hfam.
  apply ordered_selected_path_cnf_preserves_antihorn.
  apply antihorn_path_clauses_are_antihorn.
  exact Hfam.
Qed.

Definition ordered_selected_path_exhaustive_unsat_check
           (x0 : Valuation) (S : list Feature) (path : Path) : bool :=
  option_is_none
    (find_satisfying_struct_assignment
       (ordered_selected_path_cnf x0 S path)
       (powerset_atom_candidates
          (atoms_of_cnf (ordered_selected_path_cnf x0 S path)))).

Theorem ordered_selected_path_exhaustive_unsat_check_reflects_boolean_selected_path_unsat :
  forall (x0 : Valuation) (S : list Feature) (path : Path),
    ordered_selected_path_exhaustive_unsat_check x0 S path = true
    <->
    ~ boolean_path_satisfiable_with_order_under_selection x0 S path.
Proof.
  intros x0 S path.
  split.
  - intros Hcheck.
    unfold ordered_selected_path_exhaustive_unsat_check in Hcheck.
    pose proof
      (option_is_none_true
         (list Atom)
         (find_satisfying_struct_assignment
            (ordered_selected_path_cnf x0 S path)
            (powerset_atom_candidates
               (atoms_of_cnf (ordered_selected_path_cnf x0 S path))))
         Hcheck)
      as Hnone.
    pose proof
      (powerset_struct_exhaustive_solver_unsat_sound
         (ordered_selected_path_cnf x0 S path)
         Hnone)
      as Hunsat.
    intros Hbool.
    apply Hunsat.
    destruct Hbool as [rho [Hord [Hagree Hpath]]].
    exists rho.
    apply ordered_selected_path_cnf_complete_for_boolean_selected_path.
    + exact Hord.
    + exact Hagree.
    + exact Hpath.
  - intros HnotBool.
    unfold ordered_selected_path_exhaustive_unsat_check.
    set (cnf := ordered_selected_path_cnf x0 S path).
    destruct
      (find_satisfying_struct_assignment
         cnf
         (powerset_atom_candidates (atoms_of_cnf cnf)))
      as [true_atoms |] eqn:Hfind.
    + exfalso.
      apply HnotBool.
      destruct
        (find_satisfying_struct_assignment_sound
           cnf
           (powerset_atom_candidates (atoms_of_cnf cnf))
           true_atoms
           Hfind)
        as [rho Hmodel].
      unfold cnf in Hmodel.
      apply ordered_selected_path_cnf_sound_to_boolean_selected_path
        with (rho := rho).
      exact Hmodel.
    + reflexivity.
Qed.

Definition ordered_all_selected_paths_exhaustive_unsat_check
           (x0 : Valuation) (S : list Feature)
           (paths : list Path) : bool :=
  forallb (ordered_selected_path_exhaustive_unsat_check x0 S) paths.

Theorem ordered_all_selected_paths_exhaustive_unsat_check_reflects_weak_axp_boolean :
  forall (x0 : Valuation) (S : list Feature)
         (opposite_paths : list Path),
    ordered_all_selected_paths_exhaustive_unsat_check
      x0 S opposite_paths = true
    <->
    weak_axp_selected_path_condition_boolean x0 S opposite_paths.
Proof.
  intros x0 S opposite_paths.
  split.
  - intros Hcheck.
    unfold weak_axp_selected_path_condition_boolean.
    unfold boolean_opposite_paths_blocked_under_selection.
    intros path Hin.
    destruct
      (ordered_selected_path_exhaustive_unsat_check_reflects_boolean_selected_path_unsat
         x0 S path)
      as [Hcheck_to_unsat _].
    apply Hcheck_to_unsat.
    unfold ordered_all_selected_paths_exhaustive_unsat_check in Hcheck.
    apply forallb_forall with (x := path) in Hcheck.
    + exact Hcheck.
    + exact Hin.
  - intros Hweak.
    unfold ordered_all_selected_paths_exhaustive_unsat_check.
    apply forallb_forall.
    intros path Hin.
    destruct
      (ordered_selected_path_exhaustive_unsat_check_reflects_boolean_selected_path_unsat
         x0 S path)
      as [_ Hunsat_to_check].
    apply Hunsat_to_check.
    unfold weak_axp_selected_path_condition_boolean in Hweak.
    unfold boolean_opposite_paths_blocked_under_selection in Hweak.
    apply Hweak.
    exact Hin.
Qed.

Definition verified_ordered_exhaustive_weak_axp_checker
           (x0 : Valuation) (opposite_paths : list Path)
           (S : list Feature) : bool :=
  ordered_all_selected_paths_exhaustive_unsat_check x0 S opposite_paths.

Theorem verified_ordered_exhaustive_weak_axp_checker_reflects_boolean :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (S : list Feature),
    verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths S = true
    <->
    weak_axp_selected_path_condition_boolean x0 S opposite_paths.
Proof.
  intros x0 opposite_paths S.
  unfold verified_ordered_exhaustive_weak_axp_checker.
  apply ordered_all_selected_paths_exhaustive_unsat_check_reflects_weak_axp_boolean.
Qed.

Theorem verified_ordered_exhaustive_weak_axp_checker_reflects_numeric :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (S : list Feature),
    verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths S = true
    <->
    weak_axp_selected_path_condition_numeric x0 S opposite_paths.
Proof.
  intros x0 opposite_paths S.
  destruct
    (weak_axp_selected_path_condition_equivalence
       x0 S opposite_paths)
    as [Hnum_to_bool Hbool_to_num].
  split.
  - intros Hcheck.
    apply Hbool_to_num.
    apply verified_ordered_exhaustive_weak_axp_checker_reflects_boolean.
    exact Hcheck.
  - intros Hnum.
    apply verified_ordered_exhaustive_weak_axp_checker_reflects_boolean.
    apply Hnum_to_bool.
    exact Hnum.
Qed.


(* ================================================================ *)
(* 23. Concrete ordered checker plugged into deletion AXp extraction  *)
(* ================================================================ *)

(* This final section connects the executable ordered checker from Section 22
   to the generic deletion-based AXp extraction theorem from Section 12.
*)

Lemma verified_ordered_checker_reflects_weak_axp_selected_condition :
  forall (x0 : Valuation) (opposite_paths : list Path),
    bool_reflects_weak_axp_selected_condition
      (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
      x0
      opposite_paths.
Proof.
  intros x0 opposite_paths.
  unfold bool_reflects_weak_axp_selected_condition.
  intros S.
  apply verified_ordered_exhaustive_weak_axp_checker_reflects_boolean.
Qed.

Lemma verified_ordered_exhaustive_weak_axp_checker_monotone :
  forall (x0 : Valuation) (opposite_paths : list Path),
    monotone_feature_predicate
      (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths).
Proof.
  intros x0 opposite_paths.
  unfold monotone_feature_predicate.
  intros A B Hsubset HcheckA.
  apply verified_ordered_exhaustive_weak_axp_checker_reflects_boolean.
  apply weak_axp_selected_path_condition_boolean_monotone
    with (A := A).
  - exact Hsubset.
  - apply verified_ordered_exhaustive_weak_axp_checker_reflects_boolean.
    exact HcheckA.
Qed.

Theorem ordered_exhaustive_axp_extraction_returns_subset_minimal_checker_axp :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (initial : list Feature),
    verified_ordered_exhaustive_weak_axp_checker
      x0 opposite_paths initial = true ->
    subset_minimal_axp_bool
      (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial).
Proof.
  intros x0 opposite_paths initial Hinitial.
  apply deletion_extraction_returns_subset_minimal_axp.
  - apply verified_ordered_exhaustive_weak_axp_checker_monotone.
  - exact Hinitial.
Qed.

Theorem ordered_exhaustive_axp_extraction_returns_subset_minimal_weak_axp :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (initial : list Feature),
    verified_ordered_exhaustive_weak_axp_checker
      x0 opposite_paths initial = true ->
    subset_minimal_weak_axp_selected_condition_boolean
      x0
      opposite_paths
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial).
Proof.
  intros x0 opposite_paths initial Hinitial.
  apply axp_extraction_returns_subset_minimal_weak_axp_selected_condition.
  - apply verified_ordered_checker_reflects_weak_axp_selected_condition.
  - exact Hinitial.
Qed.

Theorem ordered_exhaustive_axp_extraction_returns_numeric_weak_axp :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (initial : list Feature),
    verified_ordered_exhaustive_weak_axp_checker
      x0 opposite_paths initial = true ->
    weak_axp_selected_path_condition_numeric
      x0
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial)
      opposite_paths.
Proof.
  intros x0 opposite_paths initial Hinitial.

  pose proof
    (ordered_exhaustive_axp_extraction_returns_subset_minimal_weak_axp
       x0 opposite_paths initial Hinitial)
    as HminBool.

  unfold subset_minimal_weak_axp_selected_condition_boolean in HminBool.
  destruct HminBool as [HweakBool _].

  destruct
    (weak_axp_selected_path_condition_equivalence
       x0
       (axp_extraction
          (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
          initial)
       opposite_paths)
    as [_ Hbool_to_num].

  apply Hbool_to_num.
  exact HweakBool.
Qed.


(* ================================================================ *)
(* 24. certified_horn_extraction_pipeline anti horn  *)
(* ================================================================ *)

Theorem final_certified_horn_extraction_pipeline :
  forall (x0 : Valuation)
         (tree : MDT)
         (opposite_paths : list Path)
         (initial : list Feature),
    mdt_respects_family FamilyHorn tree ->
    (forall p : Path,
        In p opposite_paths ->
        In p (paths_of_mdt tree)) ->
    verified_ordered_exhaustive_weak_axp_checker
      x0 opposite_paths initial = true ->
    weak_axp_selected_path_condition_numeric
      x0
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial)
      opposite_paths
    /\
    subset_minimal_weak_axp_selected_condition_boolean
      x0
      opposite_paths
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial)
    /\
    forall p : Path,
      In p opposite_paths ->
      all_horn_clauses
        (ordered_selected_path_cnf
           x0
           (axp_extraction
              (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
              initial)
           p) = true.
Proof.
  intros x0 tree opposite_paths initial Htree Hopposite Hcheck.
  split.
  - apply ordered_exhaustive_axp_extraction_returns_numeric_weak_axp.
    exact Hcheck.
  - split.
    + apply ordered_exhaustive_axp_extraction_returns_subset_minimal_weak_axp.
      exact Hcheck.
    + intros p Hin.
      apply ordered_selected_path_cnf_is_horn_for_horn_path.
      apply mdt_respects_family_paths with (tree := tree).
      * exact Htree.
      * apply Hopposite.
        exact Hin.
Qed.


Theorem final_certified_antihorn_extraction_pipeline :
  forall (x0 : Valuation)
         (tree : MDT)
         (opposite_paths : list Path)
         (initial : list Feature),
    mdt_respects_family FamilyAntiHorn tree ->
    (forall p : Path,
        In p opposite_paths ->
        In p (paths_of_mdt tree)) ->
    verified_ordered_exhaustive_weak_axp_checker
      x0 opposite_paths initial = true ->
    weak_axp_selected_path_condition_numeric
      x0
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial)
      opposite_paths
    /\
    subset_minimal_weak_axp_selected_condition_boolean
      x0
      opposite_paths
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial)
    /\
    forall p : Path,
      In p opposite_paths ->
      all_antihorn_clauses
        (ordered_selected_path_cnf
           x0
           (axp_extraction
              (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
              initial)
           p) = true.
Proof.
  intros x0 tree opposite_paths initial Htree Hopposite Hcheck.
  split.
  - apply ordered_exhaustive_axp_extraction_returns_numeric_weak_axp.
    exact Hcheck.
  - split.
    + apply ordered_exhaustive_axp_extraction_returns_subset_minimal_weak_axp.
      exact Hcheck.
    + intros p Hin.
      apply ordered_selected_path_cnf_is_antihorn_for_antihorn_path.
      apply mdt_respects_family_paths with (tree := tree).
      * exact Htree.
      * apply Hopposite.
        exact Hin.
Qed.

(* ================================================================ *)
(* 25. Explicit numeric subset-minimality wrapper                     *)
(* ================================================================ *)

(* The previous final pipeline already proves:
     - numeric weak-AXp validity;
     - Boolean subset-minimal weak-AXp validity.

   Since Boolean and numeric weak-AXp conditions are equivalent, this section
   packages the result as explicit numeric subset-minimality.
*)

Definition subset_minimal_weak_axp_selected_condition_numeric
           (x0 : Valuation)
           (opposite_paths : list Path)
           (S : list Feature) : Prop :=
  weak_axp_selected_path_condition_numeric x0 S opposite_paths /\
  forall T : list Feature,
    subset_features T S ->
    weak_axp_selected_path_condition_numeric x0 T opposite_paths ->
    subset_features S T.

Theorem subset_minimal_weak_axp_selected_condition_boolean_to_numeric :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (S : list Feature),
    subset_minimal_weak_axp_selected_condition_boolean
      x0 opposite_paths S ->
    subset_minimal_weak_axp_selected_condition_numeric
      x0 opposite_paths S.
Proof.
  intros x0 opposite_paths S HminBool.
  unfold subset_minimal_weak_axp_selected_condition_boolean in HminBool.
  unfold subset_minimal_weak_axp_selected_condition_numeric.
  destruct HminBool as [HweakBool HminimalBool].
  destruct
    (weak_axp_selected_path_condition_equivalence
       x0 S opposite_paths)
    as [_ Hbool_to_num].
  split.
  - apply Hbool_to_num.
    exact HweakBool.
  - intros T HTS HweakNumT.
    apply HminimalBool.
    + exact HTS.
    + destruct
        (weak_axp_selected_path_condition_equivalence
           x0 T opposite_paths)
        as [Hnum_to_bool _].
      apply Hnum_to_bool.
      exact HweakNumT.
Qed.

Theorem ordered_exhaustive_axp_extraction_returns_subset_minimal_numeric_weak_axp :
  forall (x0 : Valuation) (opposite_paths : list Path)
         (initial : list Feature),
    verified_ordered_exhaustive_weak_axp_checker
      x0 opposite_paths initial = true ->
    subset_minimal_weak_axp_selected_condition_numeric
      x0
      opposite_paths
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial).
Proof.
  intros x0 opposite_paths initial Hinitial.
  apply subset_minimal_weak_axp_selected_condition_boolean_to_numeric.
  apply ordered_exhaustive_axp_extraction_returns_subset_minimal_weak_axp.
  exact Hinitial.
Qed.

Theorem final_certified_horn_extraction_pipeline_numeric_minimal :
  forall (x0 : Valuation)
         (tree : MDT)
         (opposite_paths : list Path)
         (initial : list Feature),
    mdt_respects_family FamilyHorn tree ->
    (forall p : Path,
        In p opposite_paths ->
        In p (paths_of_mdt tree)) ->
    verified_ordered_exhaustive_weak_axp_checker
      x0 opposite_paths initial = true ->
    subset_minimal_weak_axp_selected_condition_numeric
      x0
      opposite_paths
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial)
    /\
    forall p : Path,
      In p opposite_paths ->
      all_horn_clauses
        (ordered_selected_path_cnf
           x0
           (axp_extraction
              (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
              initial)
           p) = true.
Proof.
  intros x0 tree opposite_paths initial Htree Hopposite Hcheck.
  split.
  - apply ordered_exhaustive_axp_extraction_returns_subset_minimal_numeric_weak_axp.
    exact Hcheck.
  - intros p Hin.
    apply ordered_selected_path_cnf_is_horn_for_horn_path.
    apply mdt_respects_family_paths with (tree := tree).
    + exact Htree.
    + apply Hopposite.
      exact Hin.
Qed.

Theorem final_certified_antihorn_extraction_pipeline_numeric_minimal :
  forall (x0 : Valuation)
         (tree : MDT)
         (opposite_paths : list Path)
         (initial : list Feature),
    mdt_respects_family FamilyAntiHorn tree ->
    (forall p : Path,
        In p opposite_paths ->
        In p (paths_of_mdt tree)) ->
    verified_ordered_exhaustive_weak_axp_checker
      x0 opposite_paths initial = true ->
    subset_minimal_weak_axp_selected_condition_numeric
      x0
      opposite_paths
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial)
    /\
    forall p : Path,
      In p opposite_paths ->
      all_antihorn_clauses
        (ordered_selected_path_cnf
           x0
           (axp_extraction
              (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
              initial)
           p) = true.
Proof.
  intros x0 tree opposite_paths initial Htree Hopposite Hcheck.
  split.
  - apply ordered_exhaustive_axp_extraction_returns_subset_minimal_numeric_weak_axp.
    exact Hcheck.
  - intros p Hin.
    apply ordered_selected_path_cnf_is_antihorn_for_antihorn_path.
    apply mdt_respects_family_paths with (tree := tree).
    + exact Htree.
    + apply Hopposite.
      exact Hin.
Qed.
(* ================================================================ *)
(* 26. Square2CNF theorem-compliance layer                            *)
(* ================================================================ *)

(* Square predicates use AND semantics:

      l1 /\ l2

   For a true branch, the path receives unit clauses:

      [l1] /\ [l2]

   For a false branch, the path receives the negation:

      ~(l1 /\ l2) = ~l1 \/ ~l2

   Therefore, if the square predicate has arity at most 2, every generated
   clause has size at most 2.  This is the structural theorem-compliance
   result for Square2CNF.

   This section proves the mathematical encoding/compliance layer.
   It does NOT prove the Python search implementation or runtime complexity.
*)

Definition square_pred_evalb (x : Valuation) (p : Predicate) : bool :=
  forallb (fun l => lit_evalb x l) p.

Definition square_pred_false_evalb (x : Valuation) (p : Predicate) : bool :=
  negb (square_pred_evalb x p).

Definition square_true_branch_clauses (p : Predicate) : CNF :=
  map (fun l => [encode_literal l]) p.

Definition square_false_branch_clause (p : Predicate) : Clause :=
  map (fun l => negate_signed (encode_literal l)) p.

Definition square_false_branch_clauses (p : Predicate) : CNF :=
  [square_false_branch_clause p].

Definition square_edge_evalb (x : Valuation) (e : Edge) : bool :=
  let '(p, branch) := e in
  if branch
  then square_pred_evalb x p
  else square_pred_false_evalb x p.

Definition square_edge_clauses (e : Edge) : CNF :=
  let '(p, branch) := e in
  if branch
  then square_true_branch_clauses p
  else square_false_branch_clauses p.

Fixpoint square_path_evalb (x : Valuation) (path : Path) : bool :=
  match path with
  | [] => true
  | e :: tl => andb (square_edge_evalb x e) (square_path_evalb x tl)
  end.

Fixpoint square_path_clauses (path : Path) : CNF :=
  match path with
  | [] => []
  | e :: tl => square_edge_clauses e ++ square_path_clauses tl
  end.

Lemma square_true_branch_correct :
  forall (x : Valuation) (p : Predicate),
    cnf_evalb (induced_assignment x) (square_true_branch_clauses p)
    = square_pred_evalb x p.
Proof.
  intros x p.
  unfold square_true_branch_clauses.
  unfold square_pred_evalb.
  induction p as [| l tl IH].
  - simpl. reflexivity.
  - simpl.
    rewrite encode_literal_correct.
    rewrite IH.
    destruct (lit_evalb x l); reflexivity.
Qed.

Lemma square_false_branch_clause_correct :
  forall (x : Valuation) (p : Predicate),
    clause_evalb
      (induced_assignment x)
      (square_false_branch_clause p)
    = negb (square_pred_evalb x p).
Proof.
  intros x p.
  unfold square_false_branch_clause.
  unfold square_pred_evalb.
  unfold clause_evalb.
  induction p as [| l tl IH].
  - simpl. reflexivity.
  - simpl.
    rewrite negate_signed_correct.
    rewrite encode_literal_correct.
    rewrite IH.
    destruct (lit_evalb x l);
      destruct (forallb (fun l0 : Literal => lit_evalb x l0) tl);
      reflexivity.
Qed.

Lemma square_false_branch_correct :
  forall (x : Valuation) (p : Predicate),
    cnf_evalb
      (induced_assignment x)
      (square_false_branch_clauses p)
    = square_pred_false_evalb x p.
Proof.
  intros x p.
  unfold square_false_branch_clauses.
  unfold square_pred_false_evalb.
  simpl.
  rewrite square_false_branch_clause_correct.
  destruct (square_pred_evalb x p); reflexivity.
Qed.

Theorem square_edge_encoding_correct :
  forall (x : Valuation) (e : Edge),
    cnf_evalb (induced_assignment x) (square_edge_clauses e)
    = square_edge_evalb x e.
Proof.
  intros x e.
  destruct e as [p branch].
  destruct branch.
  - simpl.
    apply square_true_branch_correct.
  - simpl.
    apply square_false_branch_correct.
Qed.

Lemma square_path_encoding_correct :
  forall (x : Valuation) (path : Path),
    cnf_evalb (induced_assignment x) (square_path_clauses path)
    = square_path_evalb x path.
Proof.
  intros x path.
  induction path as [| e tl IH].
  - simpl. reflexivity.
  - simpl.
    rewrite cnf_evalb_app.
    rewrite square_edge_encoding_correct.
    rewrite IH.
    reflexivity.
Qed.

Definition clause_at_most_two (c : Clause) : bool :=
  Nat.leb (length c) 2.

Definition cnf_at_most_two (cs : CNF) : bool :=
  forallb clause_at_most_two cs.

Definition predicate_arity_at_most_two (p : Predicate) : bool :=
  Nat.leb (length p) 2.

Lemma cnf_at_most_two_app :
  forall cs1 cs2 : CNF,
    cnf_at_most_two (cs1 ++ cs2)
    = andb (cnf_at_most_two cs1) (cnf_at_most_two cs2).
Proof.
  intros cs1 cs2.
  unfold cnf_at_most_two.
  induction cs1 as [| c tl IH].
  - simpl. reflexivity.
  - simpl.
    rewrite IH.
    destruct (clause_at_most_two c); reflexivity.
Qed.

Lemma unit_clause_at_most_two :
  forall s : SignedAtom,
    clause_at_most_two [s] = true.
Proof.
  intros s.
  unfold clause_at_most_two.
  simpl.
  reflexivity.
Qed.

Lemma structural_order_clause_at_most_two :
  forall (f : Feature) (t1 t2 : Threshold),
    clause_at_most_two (structural_order_clause f t1 t2) = true.
Proof.
  intros f t1 t2.
  unfold clause_at_most_two.
  unfold structural_order_clause.
  simpl.
  reflexivity.
Qed.

Lemma square_true_branch_clauses_at_most_two :
  forall p : Predicate,
    cnf_at_most_two (square_true_branch_clauses p) = true.
Proof.
  intros p.
  induction p as [| l tl IH].
  - simpl. reflexivity.
  - simpl.
    change
      (cnf_at_most_two (square_true_branch_clauses tl) = true).
    exact IH.
Qed.

Lemma square_false_branch_clauses_at_most_two :
  forall p : Predicate,
    predicate_arity_at_most_two p = true ->
    cnf_at_most_two (square_false_branch_clauses p) = true.
Proof.
  intros p Harity.
  unfold predicate_arity_at_most_two in Harity.
  unfold square_false_branch_clauses.
  unfold square_false_branch_clause.
  unfold cnf_at_most_two.
  simpl.
  unfold clause_at_most_two.
  rewrite length_map.
  rewrite Harity.
  reflexivity.
Qed.

Lemma square_edge_clauses_at_most_two :
  forall (p : Predicate) (branch : bool),
    predicate_arity_at_most_two p = true ->
    cnf_at_most_two (square_edge_clauses (p, branch)) = true.
Proof.
  intros p branch Harity.
  destruct branch.
  - simpl.
    apply square_true_branch_clauses_at_most_two.
  - simpl.
    apply square_false_branch_clauses_at_most_two.
    exact Harity.
Qed.

Definition square_path_2cnf_well_formed (path : Path) : Prop :=
  forall (p : Predicate) (branch : bool),
    In (p, branch) path ->
    predicate_arity_at_most_two p = true.


Theorem square_path_clauses_are_2cnf :
  forall path : Path,
    square_path_2cnf_well_formed path ->
    cnf_at_most_two (square_path_clauses path) = true.
Proof.
  intros path.
  induction path as [| e tl IH]; intros Hwf.
  - simpl. reflexivity.
  - destruct e as [p branch].
    simpl.
    rewrite cnf_at_most_two_app.
    apply andb_true_iff.
    split.
    + apply square_edge_clauses_at_most_two.
      unfold square_path_2cnf_well_formed in Hwf.
      apply (Hwf p branch).
      simpl.
      left.
      reflexivity.
    + apply IH.
      unfold square_path_2cnf_well_formed.
      intros p' branch' Hin.
      unfold square_path_2cnf_well_formed in Hwf.
      apply (Hwf p' branch').
      simpl.
      right.
      exact Hin.
Qed. 

Lemma selection_clauses_from_atoms_at_most_two :
  forall (x0 : Valuation) (S : list Feature) (atoms : list Atom),
    cnf_at_most_two (selection_clauses_from_atoms x0 S atoms) = true.
Proof.
  intros x0 S atoms.
  induction atoms as [| a tl IH].
  - simpl. reflexivity.
  - destruct a as [f t].
    simpl.
    destruct (feature_in_bool f S) eqn:Hsel.
    + simpl.
      change
        (cnf_at_most_two
           (selection_clauses_from_atoms x0 S tl) = true).
      exact IH.
    + exact IH.
Qed.


Lemma order_clauses_for_atom_at_most_two :
  forall (a : Atom) (atoms : list Atom),
    cnf_at_most_two (order_clauses_for_atom a atoms) = true.
Proof.
  intros a atoms.
  induction atoms as [| b tl IH].
  - destruct a as [f t].
    simpl. reflexivity.
  - destruct a as [f t1].
    destruct b as [g t2].
    simpl.
    destruct (andb (Nat.eqb f g) (Qle_bool t1 t2)) eqn:Htest.
    + simpl.
      change
        (cnf_at_most_two
           (order_clauses_for_atom (f, t1) tl) = true).
      exact IH.
    + exact IH.
Qed.

Lemma structural_order_clauses_aux_at_most_two :
  forall (all_atoms todo : list Atom),
    cnf_at_most_two
      (structural_order_clauses_from_atoms_aux all_atoms todo) = true.
Proof.
  intros all_atoms todo.
  induction todo as [| a tl IH].
  - simpl. reflexivity.
  - simpl.
    rewrite cnf_at_most_two_app.
    apply andb_true_iff.
    split.
    + apply order_clauses_for_atom_at_most_two.
    + exact IH.
Qed.

Theorem structural_order_clauses_at_most_two :
  forall atoms : list Atom,
    cnf_at_most_two
      (structural_order_clauses_from_atoms atoms) = true.
Proof.
  intros atoms.
  unfold structural_order_clauses_from_atoms.
  apply structural_order_clauses_aux_at_most_two.
Qed.

Definition ordered_selected_square_path_cnf
           (x0 : Valuation) (S : list Feature) (path : Path) : CNF :=
  selection_clauses_from_atoms x0 S (atoms_of_path path)
  ++ structural_order_clauses_from_atoms (atoms_of_path path)
  ++ square_path_clauses path.

Theorem ordered_selected_square_path_cnf_is_2cnf :
  forall (x0 : Valuation) (S : list Feature) (path : Path),
    square_path_2cnf_well_formed path ->
    cnf_at_most_two
      (ordered_selected_square_path_cnf x0 S path) = true.
Proof.
  intros x0 S path Hwf.
  unfold ordered_selected_square_path_cnf.
  rewrite cnf_at_most_two_app.
  apply andb_true_iff.
  split.
  - apply selection_clauses_from_atoms_at_most_two.
  - rewrite cnf_at_most_two_app.
    apply andb_true_iff.
    split.
    + apply structural_order_clauses_at_most_two.
    + apply square_path_clauses_are_2cnf.
      exact Hwf.
Qed.

Theorem square2cnf_theorem_compliance :
  forall (x0 : Valuation) (S : list Feature) (path : Path),
    square_path_2cnf_well_formed path ->
    cnf_at_most_two
      (ordered_selected_square_path_cnf x0 S path) = true
    /\
    forall x : Valuation,
      cnf_evalb (induced_assignment x) (square_path_clauses path)
      = square_path_evalb x path.
Proof.
  intros x0 S path Hwf.
  split.
  - apply ordered_selected_square_path_cnf_is_2cnf.
    exact Hwf.
  - intros x.
    apply square_path_encoding_correct.
Qed.


(* ================================================================ *)
(* 27. BEST_PER_NODE theorem-compliance layer                         *)
(* ================================================================ *)

(* BEST_PER_NODE is not itself a new logical fragment.  It is a
   per-node selection policy:

      at each node choose a single predicate from Horn or Anti-Horn.

   Therefore, the formal compliance statement is conditional on a certificate:
   every internal node must be checked as either Horn-compliant or
   AntiHorn-compliant.

   This section proves that such a certified BEST_PER_NODE tree:
     1. has every root-to-leaf path made only of Horn/AntiHorn predicates;
     2. has every ordered selected-path CNF made only of clauses that are
        Horn or AntiHorn at the clause level;
     3. still enjoys the already-proved numeric subset-minimal AXp theorem.
*)

Definition predicate_in_bestpn_bool (p : Predicate) : bool :=
  orb
    (predicate_in_family_bool FamilyHorn p)
    (predicate_in_family_bool FamilyAntiHorn p).

Definition predicate_in_bestpn (p : Predicate) : Prop :=
  predicate_in_family FamilyHorn p \/
  predicate_in_family FamilyAntiHorn p.

Lemma predicate_in_bestpn_checker_sound :
  forall p : Predicate,
    predicate_in_bestpn_bool p = true ->
    predicate_in_bestpn p.
Proof.
  intros p H.
  unfold predicate_in_bestpn_bool in H.
  unfold predicate_in_bestpn.
  apply orb_true_iff in H.
  destruct H as [Hhorn | Hanti].
  - left. exact Hhorn.
  - right. exact Hanti.
Qed.

Lemma predicate_in_bestpn_checker_complete :
  forall p : Predicate,
    predicate_in_bestpn p ->
    predicate_in_bestpn_bool p = true.
Proof.
  intros p H.
  unfold predicate_in_bestpn_bool.
  unfold predicate_in_bestpn in H.
  apply orb_true_iff.
  destruct H as [Hhorn | Hanti].
  - left. exact Hhorn.
  - right. exact Hanti.
Qed.

Fixpoint mdt_respects_bestpn_bool (tree : MDT) : bool :=
  match tree with
  | MDTLeaf _ => true
  | MDTNode p left_tree right_tree =>
      andb
        (predicate_in_bestpn_bool p)
        (andb
           (mdt_respects_bestpn_bool left_tree)
           (mdt_respects_bestpn_bool right_tree))
  end.

Fixpoint mdt_respects_bestpn (tree : MDT) : Prop :=
  match tree with
  | MDTLeaf _ => True
  | MDTNode p left_tree right_tree =>
      predicate_in_bestpn p /\
      mdt_respects_bestpn left_tree /\
      mdt_respects_bestpn right_tree
  end.

Theorem mdt_bestpn_checker_sound :
  forall tree : MDT,
    mdt_respects_bestpn_bool tree = true ->
    mdt_respects_bestpn tree.
Proof.
  intros tree.
  induction tree as [label | p left_tree IHleft right_tree IHright].
  - simpl. intros _. exact I.
  - simpl. intros H.
    apply andb_true_iff in H.
    destruct H as [Hp Hchildren].
    apply andb_true_iff in Hchildren.
    destruct Hchildren as [Hleft Hright].
    split.
    + apply predicate_in_bestpn_checker_sound.
      exact Hp.
    + split.
      * apply IHleft. exact Hleft.
      * apply IHright. exact Hright.
Qed.

Theorem mdt_bestpn_checker_complete :
  forall tree : MDT,
    mdt_respects_bestpn tree ->
    mdt_respects_bestpn_bool tree = true.
Proof.
  intros tree.
  induction tree as [label | p left_tree IHleft right_tree IHright].
  - simpl. intros _. reflexivity.
  - simpl. intros H.
    destruct H as [Hp [Hleft Hright]].
    apply andb_true_iff.
    split.
    + apply predicate_in_bestpn_checker_complete.
      exact Hp.
    + apply andb_true_iff.
      split.
      * apply IHleft. exact Hleft.
      * apply IHright. exact Hright.
Qed.

Fixpoint path_predicates_in_bestpn (path : Path) : Prop :=
  match path with
  | [] => True
  | (p, _) :: tl =>
      predicate_in_bestpn p /\
      path_predicates_in_bestpn tl
  end.

Theorem mdt_respects_bestpn_paths :
  forall (tree : MDT) (path : Path),
    mdt_respects_bestpn tree ->
    In path (paths_of_mdt tree) ->
    path_predicates_in_bestpn path.
Proof.
  intros tree.
  induction tree as [label | p left_tree IHleft right_tree IHright];
    intros path Htree Hin.
  - simpl in Hin.
    destruct Hin as [Hpath | Hbad].
    + subst path. simpl. exact I.
    + contradiction.
  - simpl in Htree.
    destruct Htree as [Hp [Hleft Hright]].
    simpl in Hin.
    apply in_app_or in Hin.
    destruct Hin as [HinLeft | HinRight].
    + apply in_map_iff in HinLeft.
      destruct HinLeft as [subpath [Heq HinSub]].
      subst path.
      simpl.
      split.
      * exact Hp.
      * apply IHleft.
        -- exact Hleft.
        -- exact HinSub.
    + apply in_map_iff in HinRight.
      destruct HinRight as [subpath [Heq HinSub]].
      subst path.
      simpl.
      split.
      * exact Hp.
      * apply IHright.
        -- exact Hright.
        -- exact HinSub.
Qed.

Definition bestpn_clause_ok (c : Clause) : Prop :=
  is_horn_clause c = true \/ is_antihorn_clause c = true.

Fixpoint all_bestpn_clauses_prop (cs : CNF) : Prop :=
  match cs with
  | [] => True
  | c :: tl =>
      bestpn_clause_ok c /\
      all_bestpn_clauses_prop tl
  end.

Lemma all_bestpn_clauses_app_intro :
  forall cs1 cs2 : CNF,
    all_bestpn_clauses_prop cs1 ->
    all_bestpn_clauses_prop cs2 ->
    all_bestpn_clauses_prop (cs1 ++ cs2).
Proof.
  intros cs1 cs2 H1 H2.
  induction cs1 as [| c tl IH].
  - simpl. exact H2.
  - simpl in H1.
    simpl.
    destruct H1 as [Hc Htl].
    split.
    + exact Hc.
    + apply IH.
      exact Htl.
Qed.

Lemma all_horn_clauses_implies_bestpn :
  forall cs : CNF,
    all_horn_clauses cs = true ->
    all_bestpn_clauses_prop cs.
Proof.
  intros cs.
  induction cs as [| c tl IH]; intros H.
  - simpl. exact I.
  - unfold all_horn_clauses in H.
    simpl in H.
    apply andb_true_iff in H.
    destruct H as [Hc Htl].
    simpl.
    split.
    + unfold bestpn_clause_ok.
      left.
      exact Hc.
    + apply IH.
      change (all_horn_clauses tl = true).
      exact Htl.
Qed.

Lemma all_antihorn_clauses_implies_bestpn :
  forall cs : CNF,
    all_antihorn_clauses cs = true ->
    all_bestpn_clauses_prop cs.
Proof.
  intros cs.
  induction cs as [| c tl IH]; intros H.
  - simpl. exact I.
  - unfold all_antihorn_clauses in H.
    simpl in H.
    apply andb_true_iff in H.
    destruct H as [Hc Htl].
    simpl.
    split.
    + unfold bestpn_clause_ok.
      right.
      exact Hc.
    + apply IH.
      change (all_antihorn_clauses tl = true).
      exact Htl.
Qed.

Lemma bestpn_edge_clauses_are_bestpn :
  forall (p : Predicate) (branch : bool),
    predicate_in_bestpn p ->
    all_bestpn_clauses_prop (edge_clauses (p, branch)).
Proof.
  intros p branch Hp.
  destruct Hp as [Hhorn | Hanti].
  - apply all_horn_clauses_implies_bestpn.
    apply horn_edge_clauses_are_horn.
    exact Hhorn.
  - apply all_antihorn_clauses_implies_bestpn.
    apply antihorn_edge_clauses_are_antihorn.
    exact Hanti.
Qed.

Theorem bestpn_path_clauses_are_bestpn :
  forall path : Path,
    path_predicates_in_bestpn path ->
    all_bestpn_clauses_prop (path_clauses path).
Proof.
  intros path.
  induction path as [| [p branch] tl IH]; intros Hpath.
  - simpl. exact I.
  - simpl in Hpath.
    destruct Hpath as [Hp Htl].
    simpl.
    apply all_bestpn_clauses_app_intro.
    + apply bestpn_edge_clauses_are_bestpn.
      exact Hp.
    + apply IH.
      exact Htl.
Qed.

Theorem checked_bestpn_mdt_paths_have_bestpn_cnf :
  forall (tree : MDT) (path : Path),
    mdt_respects_bestpn_bool tree = true ->
    In path (paths_of_mdt tree) ->
    all_bestpn_clauses_prop (path_clauses path).
Proof.
  intros tree path Hcheck HinPath.
  apply bestpn_path_clauses_are_bestpn.
  apply mdt_respects_bestpn_paths with (tree := tree).
  - apply mdt_bestpn_checker_sound.
    exact Hcheck.
  - exact HinPath.
Qed.

Theorem ordered_selected_bestpn_path_cnf_has_bestpn_clauses :
  forall (x0 : Valuation) (S : list Feature) (path : Path),
    path_predicates_in_bestpn path ->
    all_bestpn_clauses_prop
      (ordered_selected_path_cnf x0 S path).
Proof.
  intros x0 S path Hpath.
  unfold ordered_selected_path_cnf.
  apply all_bestpn_clauses_app_intro.
  - apply all_horn_clauses_implies_bestpn.
    apply ordered_selection_clauses_from_atoms_are_horn.
  - apply all_bestpn_clauses_app_intro.
    + apply all_horn_clauses_implies_bestpn.
      apply structural_order_clauses_are_horn.
    + apply bestpn_path_clauses_are_bestpn.
      exact Hpath.
Qed.

Theorem checked_bestpn_mdt_ordered_selected_paths_have_bestpn_cnf :
  forall (x0 : Valuation) (S : list Feature)
         (tree : MDT) (path : Path),
    mdt_respects_bestpn_bool tree = true ->
    In path (paths_of_mdt tree) ->
    all_bestpn_clauses_prop
      (ordered_selected_path_cnf x0 S path).
Proof.
  intros x0 S tree path Hcheck HinPath.
  apply ordered_selected_bestpn_path_cnf_has_bestpn_clauses.
  apply mdt_respects_bestpn_paths with (tree := tree).
  - apply mdt_bestpn_checker_sound.
    exact Hcheck.
  - exact HinPath.
Qed.

Theorem final_certified_bestpn_extraction_pipeline_numeric_minimal :
  forall (x0 : Valuation)
         (tree : MDT)
         (opposite_paths : list Path)
         (initial : list Feature),
    mdt_respects_bestpn_bool tree = true ->
    (forall p : Path,
        In p opposite_paths ->
        In p (paths_of_mdt tree)) ->
    verified_ordered_exhaustive_weak_axp_checker
      x0 opposite_paths initial = true ->
    subset_minimal_weak_axp_selected_condition_numeric
      x0
      opposite_paths
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial)
    /\
    forall p : Path,
      In p opposite_paths ->
      all_bestpn_clauses_prop
        (ordered_selected_path_cnf
           x0
           (axp_extraction
              (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
              initial)
           p).
Proof.
  intros x0 tree opposite_paths initial Htree Hopposite Hcheck.
  split.
  - apply ordered_exhaustive_axp_extraction_returns_subset_minimal_numeric_weak_axp.
    exact Hcheck.
  - intros p Hin.
    apply checked_bestpn_mdt_ordered_selected_paths_have_bestpn_cnf
      with (tree := tree).
    + exact Htree.
    + apply Hopposite.
      exact Hin.
Qed.


(* ================================================================ *)
(*                               path-level safety checker,*)
(* ================================================================ *)

Definition polynomial_safe_ordered_path_cnf
           (x0 : Valuation) (S : list Feature) (path : Path) : Prop :=
  all_horn_clauses
    (ordered_selected_path_cnf x0 S path) = true
  \/
  all_antihorn_clauses
    (ordered_selected_path_cnf x0 S path) = true
  \/
  cnf_at_most_two
    (ordered_selected_path_cnf x0 S path) = true.
    
Definition mixed_bestpn_paths_polynomial_safe
           (x0 : Valuation) (S : list Feature)
           (paths : list Path) : Prop :=
  forall p : Path,
    In p paths ->
    polynomial_safe_ordered_path_cnf x0 S p.

Theorem mixed_bestpn_axp_is_numeric_subset_minimal_under_polynomial_certificate :
  forall (x0 : Valuation)
         (opposite_paths : list Path)
         (initial : list Feature),
    mixed_bestpn_paths_polynomial_safe x0 initial opposite_paths ->
    verified_ordered_exhaustive_weak_axp_checker
      x0 opposite_paths initial = true ->
    subset_minimal_weak_axp_selected_condition_numeric
      x0
      opposite_paths
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial).
Proof.
  intros x0 opposite_paths initial Hpoly Hcheck.
  apply ordered_exhaustive_axp_extraction_returns_subset_minimal_numeric_weak_axp.
  exact Hcheck.
Qed.


Theorem mixed_bestpn_axp_and_polynomial_certificate :
  forall (x0 : Valuation)
         (opposite_paths : list Path)
         (initial : list Feature),
    mixed_bestpn_paths_polynomial_safe x0 initial opposite_paths ->
    verified_ordered_exhaustive_weak_axp_checker
      x0 opposite_paths initial = true ->
    mixed_bestpn_paths_polynomial_safe x0 initial opposite_paths
    /\
    subset_minimal_weak_axp_selected_condition_numeric
      x0
      opposite_paths
      (axp_extraction
         (verified_ordered_exhaustive_weak_axp_checker x0 opposite_paths)
         initial).
Proof.
  intros x0 opposite_paths initial Hpoly Hcheck.
  split.
  - exact Hpoly.
  - apply ordered_exhaustive_axp_extraction_returns_subset_minimal_numeric_weak_axp.
    exact Hcheck.
Qed.
