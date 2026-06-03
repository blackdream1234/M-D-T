(*
  GSNH_Affine_AXp.v

  Affine / XOR proof layer for GSNH-MDT.

  Scope:
  - Boolean affine equations over already-encoded threshold atoms.
  - An affine equation is:
        x1 xor x2 xor ... xor xn = b
  - False branch is encoded by flipping the right-hand side.
  - This proves affine path encoding correctness.
  - This does NOT yet prove the Gaussian-elimination solver.
  - This does NOT yet combine affine constraints with structural threshold-order clauses.
*)

Require Import Coq.Lists.List.
Require Import Coq.Bool.Bool.
Require Import Coq.QArith.QArith.
Require Import GSNH_Threshold_AXp.

Import ListNotations.
Open Scope Q_scope.

(* ================================================================ *)
(* 1. Affine equations over signed Boolean atoms                     *)
(* ================================================================ *)

Definition xor_list (bs : list bool) : bool :=
  fold_right xorb false bs.

Record AffineEquation := {
  ae_terms : list SignedAtom;
  ae_rhs   : bool
}.

Definition affine_lhs_evalb
           (rho : Assignment)
           (terms : list SignedAtom) : bool :=
  xor_list (map (signed_evalb rho) terms).

Definition affine_equation_evalb
           (rho : Assignment)
           (e : AffineEquation) : bool :=
  Bool.eqb (affine_lhs_evalb rho e.(ae_terms)) e.(ae_rhs).

Definition affine_complement
           (e : AffineEquation) : AffineEquation :=
  {|
    ae_terms := e.(ae_terms);
    ae_rhs   := negb e.(ae_rhs)
  |}.

Lemma bool_eqb_negb_r :
  forall a b : bool,
    Bool.eqb a (negb b) = negb (Bool.eqb a b).
Proof.
  intros a b.
  destruct a, b; reflexivity.
Qed.

Theorem affine_complement_correct :
  forall (rho : Assignment) (e : AffineEquation),
    affine_equation_evalb rho (affine_complement e)
    = negb (affine_equation_evalb rho e).
Proof.
  intros rho e.
  unfold affine_equation_evalb, affine_complement.
  simpl.
  rewrite bool_eqb_negb_r.
  reflexivity.
Qed.

(* ================================================================ *)
(* 2. Affine edges and paths                                         *)
(* ================================================================ *)

(* Branch convention:
   true  = affine equation is satisfied
   false = complement equation is satisfied
*)
Definition AffineEdge := (AffineEquation * bool)%type.
Definition AffinePath := list AffineEdge.

Definition affine_edge_equation
           (e : AffineEdge) : AffineEquation :=
  let '(eqn, branch) := e in
  if branch then eqn else affine_complement eqn.

Definition affine_edge_evalb
           (rho : Assignment)
           (e : AffineEdge) : bool :=
  affine_equation_evalb rho (affine_edge_equation e).

Fixpoint affine_path_evalb
         (rho : Assignment)
         (path : AffinePath) : bool :=
  match path with
  | [] => true
  | e :: tl =>
      andb (affine_edge_evalb rho e)
           (affine_path_evalb rho tl)
  end.

Definition affine_path_equations
           (path : AffinePath) : list AffineEquation :=
  map affine_edge_equation path.

Definition affine_system_evalb
           (rho : Assignment)
           (eqs : list AffineEquation) : bool :=
  forallb (affine_equation_evalb rho) eqs.

Theorem affine_false_branch_is_complement :
  forall (rho : Assignment) (eqn : AffineEquation),
    affine_edge_evalb rho (eqn, false)
    = negb (affine_equation_evalb rho eqn).
Proof.
  intros rho eqn.
  unfold affine_edge_evalb, affine_edge_equation.
  simpl.
  apply affine_complement_correct.
Qed.

Theorem affine_path_encoding_correct :
  forall (rho : Assignment) (path : AffinePath),
    affine_system_evalb rho (affine_path_equations path)
    = affine_path_evalb rho path.
Proof.
  intros rho path.
  induction path as [| e tl IH].
  - reflexivity.
  - simpl.
    rewrite IH.
    reflexivity.
Qed.

(* ================================================================ *)
(* 3. Affine weak-AXp opposite-path check                            *)
(* ================================================================ *)

Definition affine_weak_axp_direct_evalb
           (rho : Assignment)
           (opposite_paths : list AffinePath) : bool :=
  forallb
    (fun p => negb (affine_path_evalb rho p))
    opposite_paths.

Definition affine_weak_axp_encoded_evalb
           (rho : Assignment)
           (opposite_paths : list AffinePath) : bool :=
  forallb
    (fun p =>
       negb
         (affine_system_evalb rho (affine_path_equations p)))
    opposite_paths.

Theorem affine_weak_axp_encoding_correct :
  forall (rho : Assignment) (opposite_paths : list AffinePath),
    affine_weak_axp_encoded_evalb rho opposite_paths
    = affine_weak_axp_direct_evalb rho opposite_paths.
Proof.
  intros rho opposite_paths.
  induction opposite_paths as [| p tl IH].
  - reflexivity.
  - simpl.
    rewrite affine_path_encoding_correct.
    rewrite IH.
    reflexivity.
Qed.

(* ================================================================ *)
(* 4. Small sanity example                                           *)
(* ================================================================ *)

Definition affine_atom_0 : SignedAtom :=
  {| sa_atom := (0%nat, 0); sa_positive := true |}.

Definition affine_atom_1 : SignedAtom :=
  {| sa_atom := (1%nat, 0); sa_positive := true |}.

Definition affine_xor_example : AffineEquation :=
  {|
    ae_terms := [affine_atom_0; affine_atom_1];
    ae_rhs := false
  |}.

Definition affine_sample_assignment : Assignment :=
  fun a =>
    match a with
    | (0%nat, _) => true
    | (1%nat, _) => true
    | _ => false
    end.

Example affine_xor_example_true :
  affine_equation_evalb affine_sample_assignment affine_xor_example = true.
Proof.
  reflexivity.
Qed.

Example affine_xor_example_false_branch :
  affine_edge_evalb affine_sample_assignment (affine_xor_example, false)
  = false.
Proof.
  reflexivity.
Qed.

(* ================================================================ *)
(* 5. Selected-feature agreement for affine atoms                    *)
(* ================================================================ *)

Definition affine_atoms_of_equation (e : AffineEquation) : list Atom :=
  map sa_atom e.(ae_terms).

Definition affine_atoms_of_system (eqs : list AffineEquation) : list Atom :=
  flat_map affine_atoms_of_equation eqs.

Definition affine_selected_atom_agreement
           (x0 : Valuation)
           (S : list Feature)
           (atoms : list Atom)
           (rho : Assignment) : Prop :=
  forall (f : Feature) (t : Threshold),
    In (f,t) atoms ->
    In f S ->
    induced_assignment x0 (f,t) = rho (f,t).

Definition affine_system_satisfiable_under_selection
           (x0 : Valuation)
           (S : list Feature)
           (eqs : list AffineEquation) : Prop :=
  exists rho : Assignment,
    affine_selected_atom_agreement
      x0 S (affine_atoms_of_system eqs) rho
    /\
    affine_system_evalb rho eqs = true.

Definition affine_path_satisfiable_under_selection
           (x0 : Valuation)
           (S : list Feature)
           (path : AffinePath) : Prop :=
  exists rho : Assignment,
    affine_selected_atom_agreement
      x0 S (affine_atoms_of_system (affine_path_equations path)) rho
    /\
    affine_path_evalb rho path = true.

Theorem affine_path_system_satisfiable_equiv :
  forall (x0 : Valuation) (S : list Feature) (path : AffinePath),
    affine_system_satisfiable_under_selection
      x0 S (affine_path_equations path)
    <->
    affine_path_satisfiable_under_selection x0 S path.
Proof.
  intros x0 S path.
  split.
  - intros [rho [Hagree Hsys]].
    exists rho.
    split.
    + exact Hagree.
    + rewrite <- affine_path_encoding_correct.
      exact Hsys.
  - intros [rho [Hagree Hpath]].
    exists rho.
    split.
    + exact Hagree.
    + rewrite affine_path_encoding_correct.
      exact Hpath.
Qed.

(* ================================================================ *)
(* 6. Prop-level affine weak-AXp condition                           *)
(* ================================================================ *)

Definition affine_opposite_paths_blocked_under_selection
           (x0 : Valuation)
           (S : list Feature)
           (opposite_paths : list AffinePath) : Prop :=
  forall p : AffinePath,
    In p opposite_paths ->
    ~ affine_path_satisfiable_under_selection x0 S p.

Definition affine_weak_axp_selected_path_condition
           (x0 : Valuation)
           (S : list Feature)
           (opposite_paths : list AffinePath) : Prop :=
  affine_opposite_paths_blocked_under_selection x0 S opposite_paths.

Definition affine_system_paths_blocked_under_selection
           (x0 : Valuation)
           (S : list Feature)
           (opposite_paths : list AffinePath) : Prop :=
  forall p : AffinePath,
    In p opposite_paths ->
    ~ affine_system_satisfiable_under_selection
        x0 S (affine_path_equations p).

Theorem affine_weak_axp_path_system_equiv :
  forall (x0 : Valuation)
         (S : list Feature)
         (opposite_paths : list AffinePath),
    affine_system_paths_blocked_under_selection x0 S opposite_paths
    <->
    affine_weak_axp_selected_path_condition x0 S opposite_paths.
Proof.
  intros x0 S opposite_paths.
  split.
  - intros Hsys.
    unfold affine_weak_axp_selected_path_condition.
    unfold affine_opposite_paths_blocked_under_selection.
    intros p Hin HsatPath.
    apply (Hsys p Hin).
    apply affine_path_system_satisfiable_equiv.
    exact HsatPath.
  - intros Hpath.
    unfold affine_system_paths_blocked_under_selection.
    intros p Hin HsatSys.
    unfold affine_weak_axp_selected_path_condition in Hpath.
    unfold affine_opposite_paths_blocked_under_selection in Hpath.
    apply (Hpath p Hin).
    apply affine_path_system_satisfiable_equiv.
    exact HsatSys.
Qed.

