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

(* ================================================================ *)
(* 7. Exhaustive finite affine solver specification                  *)
(* ================================================================ *)

(* This is an executable reference solver for affine systems.
   It enumerates all subsets of a finite atom list and interprets each
   subset as the atoms assigned true.

   This is NOT the final polynomial GF(2) Gaussian-elimination solver.
   It is a small certified reference checker used to state and test the
   affine satisfiability semantics.
*)

Definition affine_atom_eqb (a b : Atom) : bool :=
  let '(fa, ta) := a in
  let '(fb, tb) := b in
  andb (Nat.eqb fa fb) (Qeq_bool ta tb).

Fixpoint affine_atom_in_bool (a : Atom) (atoms : list Atom) : bool :=
  match atoms with
  | [] => false
  | b :: tl =>
      if affine_atom_eqb a b
      then true
      else affine_atom_in_bool a tl
  end.

Definition affine_assignment_from_true_atoms
           (true_atoms : list Atom) : Assignment :=
  fun a => affine_atom_in_bool a true_atoms.

Fixpoint affine_powerset_atoms (atoms : list Atom) : list (list Atom) :=
  match atoms with
  | [] => [[]]
  | a :: tl =>
      let ps := affine_powerset_atoms tl in
      ps ++ map (fun xs => a :: xs) ps
  end.

Fixpoint find_affine_assignment_from_candidates
         (eqs : list AffineEquation)
         (candidates : list (list Atom)) : option (list Atom) :=
  match candidates with
  | [] => None
  | true_atoms :: tl =>
      if affine_system_evalb
           (affine_assignment_from_true_atoms true_atoms)
           eqs
      then Some true_atoms
      else find_affine_assignment_from_candidates eqs tl
  end.

Definition find_affine_satisfying_assignment
           (eqs : list AffineEquation)
           (atoms : list Atom) : option (list Atom) :=
  find_affine_assignment_from_candidates
    eqs
    (affine_powerset_atoms atoms).

Theorem find_affine_assignment_from_candidates_sound :
  forall (eqs : list AffineEquation)
         (candidates : list (list Atom))
         (true_atoms : list Atom),
    find_affine_assignment_from_candidates eqs candidates
      = Some true_atoms ->
    affine_system_evalb
      (affine_assignment_from_true_atoms true_atoms)
      eqs = true.
Proof.
  intros eqs candidates.
  induction candidates as [| cand tl IH]; intros true_atoms Hfind.
  - simpl in Hfind.
    discriminate.
  - simpl in Hfind.
    destruct
      (affine_system_evalb
         (affine_assignment_from_true_atoms cand)
         eqs) eqn:Hcand.
    + inversion Hfind.
      subst true_atoms.
      exact Hcand.
    + apply IH.
      exact Hfind.
Qed.

Theorem find_affine_satisfying_assignment_sound :
  forall (eqs : list AffineEquation)
         (atoms true_atoms : list Atom),
    find_affine_satisfying_assignment eqs atoms = Some true_atoms ->
    exists rho : Assignment,
      affine_system_evalb rho eqs = true.
Proof.
  intros eqs atoms true_atoms Hfind.
  exists (affine_assignment_from_true_atoms true_atoms).
  unfold find_affine_satisfying_assignment in Hfind.
  eapply find_affine_assignment_from_candidates_sound.
  exact Hfind.
Qed.

Theorem find_affine_assignment_from_candidates_complete_candidate :
  forall (eqs : list AffineEquation)
         (candidates : list (list Atom))
         (witness : list Atom),
    In witness candidates ->
    affine_system_evalb
      (affine_assignment_from_true_atoms witness)
      eqs = true ->
    exists found : list Atom,
      find_affine_assignment_from_candidates eqs candidates
        = Some found.
Proof.
  intros eqs candidates.
  induction candidates as [| cand tl IH]; intros witness Hin Hwit.
  - contradiction.
  - simpl in Hin.
    simpl.
    destruct
      (affine_system_evalb
         (affine_assignment_from_true_atoms cand)
         eqs) eqn:Hcand.
    + exists cand.
      reflexivity.
    + destruct Hin as [Heq | HinTl].
      * subst witness.
        rewrite Hwit in Hcand.
        discriminate.
      * apply (IH witness).
        -- exact HinTl.
        -- exact Hwit.
Qed.

Theorem find_affine_satisfying_assignment_complete_candidate :
  forall (eqs : list AffineEquation)
         (atoms witness : list Atom),
    In witness (affine_powerset_atoms atoms) ->
    affine_system_evalb
      (affine_assignment_from_true_atoms witness)
      eqs = true ->
    exists found : list Atom,
      find_affine_satisfying_assignment eqs atoms = Some found.
Proof.
  intros eqs atoms witness Hin Hwit.
  unfold find_affine_satisfying_assignment.
  apply find_affine_assignment_from_candidates_complete_candidate
    with (witness := witness).
  - exact Hin.
  - exact Hwit.
Qed.

Definition affine_option_is_none {A : Type} (o : option A) : bool :=
  match o with
  | None => true
  | Some _ => false
  end.

Definition affine_exhaustive_unsat_check
           (eqs : list AffineEquation)
           (atoms : list Atom) : bool :=
  affine_option_is_none
    (find_affine_satisfying_assignment eqs atoms).

Lemma find_affine_assignment_from_candidates_none_unsat :
  forall (eqs : list AffineEquation)
         (candidates : list (list Atom)),
    find_affine_assignment_from_candidates eqs candidates = None ->
    forall true_atoms : list Atom,
      In true_atoms candidates ->
      affine_system_evalb
        (affine_assignment_from_true_atoms true_atoms)
        eqs = false.
Proof.
  intros eqs candidates.
  induction candidates as [| cand tl IH]; intros Hnone true_atoms Hin.
  - contradiction.
  - simpl in Hnone.
    destruct
      (affine_system_evalb
         (affine_assignment_from_true_atoms cand)
         eqs) eqn:Hcand.
    + discriminate.
    + simpl in Hin.
      destruct Hin as [Heq | HinTl].
      * subst true_atoms.
        exact Hcand.
      * apply IH.
        -- exact Hnone.
        -- exact HinTl.
Qed.

Theorem affine_exhaustive_unsat_check_sound_candidates :
  forall (eqs : list AffineEquation)
         (atoms true_atoms : list Atom),
    affine_exhaustive_unsat_check eqs atoms = true ->
    In true_atoms (affine_powerset_atoms atoms) ->
    affine_system_evalb
      (affine_assignment_from_true_atoms true_atoms)
      eqs = false.
Proof.
  intros eqs atoms true_atoms Hcheck Hin.
  unfold affine_exhaustive_unsat_check in Hcheck.
  unfold find_affine_satisfying_assignment in Hcheck.
  destruct
    (find_affine_assignment_from_candidates
       eqs
       (affine_powerset_atoms atoms)) eqn:Hfind.
  - simpl in Hcheck.
    discriminate.
  - eapply find_affine_assignment_from_candidates_none_unsat.
    + exact Hfind.
    + exact Hin.
Qed.

(* ================================================================ *)
(* 8. Path-level exhaustive affine unsat checker                     *)
(* ================================================================ *)

Definition affine_path_exhaustive_unsat_check
           (path : AffinePath) : bool :=
  affine_exhaustive_unsat_check
    (affine_path_equations path)
    (affine_atoms_of_system (affine_path_equations path)).

Theorem affine_path_exhaustive_unsat_check_sound_candidates :
  forall (path : AffinePath) (true_atoms : list Atom),
    affine_path_exhaustive_unsat_check path = true ->
    In true_atoms
       (affine_powerset_atoms
          (affine_atoms_of_system (affine_path_equations path))) ->
    affine_system_evalb
      (affine_assignment_from_true_atoms true_atoms)
      (affine_path_equations path) = false.
Proof.
  intros path true_atoms Hcheck Hin.
  unfold affine_path_exhaustive_unsat_check in Hcheck.
  eapply affine_exhaustive_unsat_check_sound_candidates.
  - exact Hcheck.
  - exact Hin.
Qed.

Theorem affine_path_exhaustive_unsat_check_sound_path_candidates :
  forall (path : AffinePath) (true_atoms : list Atom),
    affine_path_exhaustive_unsat_check path = true ->
    In true_atoms
       (affine_powerset_atoms
          (affine_atoms_of_system (affine_path_equations path))) ->
    affine_path_evalb
      (affine_assignment_from_true_atoms true_atoms)
      path = false.
Proof.
  intros path true_atoms Hcheck Hin.
  pose proof
    (affine_path_exhaustive_unsat_check_sound_candidates
       path true_atoms Hcheck Hin)
    as HsysFalse.
  rewrite
    (affine_path_encoding_correct
       (affine_assignment_from_true_atoms true_atoms)
       path)
    in HsysFalse.
  exact HsysFalse.
Qed.

(* ================================================================ *)
(* 9. All-opposite-path affine exhaustive checker                    *)
(* ================================================================ *)

Definition affine_all_paths_exhaustive_unsat_check
           (paths : list AffinePath) : bool :=
  forallb affine_path_exhaustive_unsat_check paths.

Definition affine_candidate_opposite_paths_blocked
           (opposite_paths : list AffinePath) : Prop :=
  forall (p : AffinePath) (true_atoms : list Atom),
    In p opposite_paths ->
    In true_atoms
       (affine_powerset_atoms
          (affine_atoms_of_system (affine_path_equations p))) ->
    affine_path_evalb
      (affine_assignment_from_true_atoms true_atoms)
      p = false.

Theorem affine_all_paths_exhaustive_unsat_check_sound_candidates :
  forall (opposite_paths : list AffinePath),
    affine_all_paths_exhaustive_unsat_check opposite_paths = true ->
    affine_candidate_opposite_paths_blocked opposite_paths.
Proof.
  intros opposite_paths Hcheck.
  unfold affine_candidate_opposite_paths_blocked.
  intros p true_atoms Hin HinCand.
  unfold affine_all_paths_exhaustive_unsat_check in Hcheck.
  apply forallb_forall with (x := p) in Hcheck.
  - eapply affine_path_exhaustive_unsat_check_sound_path_candidates.
    + exact Hcheck.
    + exact HinCand.
  - exact Hin.
Qed.

(* ================================================================ *)
(* 10. Affine exhaustive checker under selected-feature constraints  *)
(* ================================================================ *)

(* AXp semantics requires selected features to keep their original
   truth values from x0. This section adds that constraint to the
   finite exhaustive affine checker.

   The checker still works over finite candidate assignments.
   It is still not the final polynomial GF(2) solver.
*)

Fixpoint affine_feature_in_bool
         (f : Feature) (S : list Feature) : bool :=
  match S with
  | [] => false
  | g :: tl =>
      if Nat.eqb f g
      then true
      else affine_feature_in_bool f tl
  end.

Definition affine_selected_candidate_agreementb
           (x0 : Valuation)
           (S : list Feature)
           (atoms : list Atom)
           (true_atoms : list Atom) : bool :=
  forallb
    (fun a =>
       let '(f,t) := a in
       if affine_feature_in_bool f S
       then
         Bool.eqb
           (affine_assignment_from_true_atoms true_atoms (f,t))
           (induced_assignment x0 (f,t))
       else true)
    atoms.

Definition affine_system_evalb_under_selection
           (x0 : Valuation)
           (S : list Feature)
           (atoms : list Atom)
           (eqs : list AffineEquation)
           (true_atoms : list Atom) : bool :=
  andb
    (affine_selected_candidate_agreementb x0 S atoms true_atoms)
    (affine_system_evalb
       (affine_assignment_from_true_atoms true_atoms)
       eqs).

Fixpoint find_affine_assignment_under_selection_from_candidates
         (x0 : Valuation)
         (S : list Feature)
         (atoms : list Atom)
         (eqs : list AffineEquation)
         (candidates : list (list Atom)) : option (list Atom) :=
  match candidates with
  | [] => None
  | true_atoms :: tl =>
      if affine_system_evalb_under_selection x0 S atoms eqs true_atoms
      then Some true_atoms
      else
        find_affine_assignment_under_selection_from_candidates
          x0 S atoms eqs tl
  end.

Definition find_affine_satisfying_assignment_under_selection
           (x0 : Valuation)
           (S : list Feature)
           (eqs : list AffineEquation)
           (atoms : list Atom) : option (list Atom) :=
  find_affine_assignment_under_selection_from_candidates
    x0 S atoms eqs (affine_powerset_atoms atoms).

Definition affine_exhaustive_unsat_check_under_selection
           (x0 : Valuation)
           (S : list Feature)
           (eqs : list AffineEquation)
           (atoms : list Atom) : bool :=
  affine_option_is_none
    (find_affine_satisfying_assignment_under_selection
       x0 S eqs atoms).

Theorem find_affine_assignment_under_selection_from_candidates_sound :
  forall (x0 : Valuation)
         (S : list Feature)
         (atoms : list Atom)
         (eqs : list AffineEquation)
         (candidates : list (list Atom))
         (true_atoms : list Atom),
    find_affine_assignment_under_selection_from_candidates
      x0 S atoms eqs candidates = Some true_atoms ->
    affine_system_evalb_under_selection
      x0 S atoms eqs true_atoms = true.
Proof.
  intros x0 S atoms eqs candidates.
  induction candidates as [| cand tl IH]; intros true_atoms Hfind.
  - simpl in Hfind.
    discriminate.
  - simpl in Hfind.
    destruct
      (affine_system_evalb_under_selection
         x0 S atoms eqs cand) eqn:Hcand.
    + inversion Hfind.
      subst true_atoms.
      exact Hcand.
    + apply IH.
      exact Hfind.
Qed.

Theorem find_affine_satisfying_assignment_under_selection_sound :
  forall (x0 : Valuation)
         (S : list Feature)
         (eqs : list AffineEquation)
         (atoms true_atoms : list Atom),
    find_affine_satisfying_assignment_under_selection
      x0 S eqs atoms = Some true_atoms ->
    affine_selected_candidate_agreementb
      x0 S atoms true_atoms = true
    /\
    affine_system_evalb
      (affine_assignment_from_true_atoms true_atoms)
      eqs = true.
Proof.
  intros x0 S eqs atoms true_atoms Hfind.
  unfold find_affine_satisfying_assignment_under_selection in Hfind.
  pose proof
    (find_affine_assignment_under_selection_from_candidates_sound
       x0 S atoms eqs
       (affine_powerset_atoms atoms)
       true_atoms
       Hfind)
    as Hunder.
  unfold affine_system_evalb_under_selection in Hunder.
  apply andb_true_iff in Hunder.
  exact Hunder.
Qed.

Lemma find_affine_assignment_under_selection_from_candidates_none_unsat :
  forall (x0 : Valuation)
         (S : list Feature)
         (atoms : list Atom)
         (eqs : list AffineEquation)
         (candidates : list (list Atom)),
    find_affine_assignment_under_selection_from_candidates
      x0 S atoms eqs candidates = None ->
    forall true_atoms : list Atom,
      In true_atoms candidates ->
      affine_system_evalb_under_selection
        x0 S atoms eqs true_atoms = false.
Proof.
  intros x0 S atoms eqs candidates.
  induction candidates as [| cand tl IH]; intros Hnone true_atoms Hin.
  - contradiction.
  - simpl in Hnone.
    destruct
      (affine_system_evalb_under_selection
         x0 S atoms eqs cand) eqn:Hcand.
    + discriminate.
    + simpl in Hin.
      destruct Hin as [Heq | HinTl].
      * subst true_atoms.
        exact Hcand.
      * apply IH.
        -- exact Hnone.
        -- exact HinTl.
Qed.

Theorem affine_exhaustive_unsat_check_under_selection_sound_candidates :
  forall (x0 : Valuation)
         (S : list Feature)
         (eqs : list AffineEquation)
         (atoms true_atoms : list Atom),
    affine_exhaustive_unsat_check_under_selection
      x0 S eqs atoms = true ->
    In true_atoms (affine_powerset_atoms atoms) ->
    affine_system_evalb_under_selection
      x0 S atoms eqs true_atoms = false.
Proof.
  intros x0 S eqs atoms true_atoms Hcheck Hin.
  unfold affine_exhaustive_unsat_check_under_selection in Hcheck.
  unfold find_affine_satisfying_assignment_under_selection in Hcheck.
  destruct
    (find_affine_assignment_under_selection_from_candidates
       x0 S atoms eqs (affine_powerset_atoms atoms)) eqn:Hfind.
  - simpl in Hcheck.
    discriminate.
  - eapply find_affine_assignment_under_selection_from_candidates_none_unsat.
    + exact Hfind.
    + exact Hin.
Qed.

(* ================================================================ *)
(* 11. Path-level affine UNSAT under selected-feature constraints    *)
(* ================================================================ *)

Definition affine_selected_candidate_path_evalb
           (x0 : Valuation)
           (S : list Feature)
           (path : AffinePath)
           (true_atoms : list Atom) : bool :=
  let eqs := affine_path_equations path in
  let atoms := affine_atoms_of_system eqs in
  andb
    (affine_selected_candidate_agreementb
       x0 S atoms true_atoms)
    (affine_path_evalb
       (affine_assignment_from_true_atoms true_atoms)
       path).

Definition affine_path_exhaustive_unsat_check_under_selection
           (x0 : Valuation)
           (S : list Feature)
           (path : AffinePath) : bool :=
  let eqs := affine_path_equations path in
  let atoms := affine_atoms_of_system eqs in
  affine_exhaustive_unsat_check_under_selection
    x0 S eqs atoms.

Theorem affine_path_exhaustive_unsat_check_under_selection_sound_candidates :
  forall (x0 : Valuation)
         (S : list Feature)
         (path : AffinePath)
         (true_atoms : list Atom),
    affine_path_exhaustive_unsat_check_under_selection
      x0 S path = true ->
    In true_atoms
       (affine_powerset_atoms
          (affine_atoms_of_system (affine_path_equations path))) ->
    affine_selected_candidate_path_evalb
      x0 S path true_atoms = false.
Proof.
  intros x0 S path true_atoms Hcheck Hin.
  unfold affine_path_exhaustive_unsat_check_under_selection in Hcheck.
  pose proof
    (affine_exhaustive_unsat_check_under_selection_sound_candidates
       x0 S
       (affine_path_equations path)
       (affine_atoms_of_system (affine_path_equations path))
       true_atoms
       Hcheck
       Hin)
    as HunderFalse.
  unfold affine_system_evalb_under_selection in HunderFalse.
  unfold affine_selected_candidate_path_evalb.
  rewrite
    (affine_path_encoding_correct
       (affine_assignment_from_true_atoms true_atoms)
       path)
    in HunderFalse.
  exact HunderFalse.
Qed.
(* ================================================================ *)
(* 12. All affine opposite paths blocked under selection             *)
(* ================================================================ *)

Definition affine_all_paths_exhaustive_unsat_check_under_selection
           (x0 : Valuation)
           (S : list Feature)
           (paths : list AffinePath) : bool :=
  forallb
    (affine_path_exhaustive_unsat_check_under_selection x0 S)
    paths.

Definition affine_candidate_opposite_paths_blocked_under_selection
           (x0 : Valuation)
           (S : list Feature)
           (opposite_paths : list AffinePath) : Prop :=
  forall (p : AffinePath) (true_atoms : list Atom),
    In p opposite_paths ->
    In true_atoms
       (affine_powerset_atoms
          (affine_atoms_of_system (affine_path_equations p))) ->
    affine_selected_candidate_path_evalb
      x0 S p true_atoms = false.

Theorem affine_all_paths_exhaustive_unsat_check_under_selection_sound_candidates :
  forall (x0 : Valuation)
         (S : list Feature)
         (opposite_paths : list AffinePath),
    affine_all_paths_exhaustive_unsat_check_under_selection
      x0 S opposite_paths = true ->
    affine_candidate_opposite_paths_blocked_under_selection
      x0 S opposite_paths.
Proof.
  intros x0 S opposite_paths Hcheck.
  unfold affine_candidate_opposite_paths_blocked_under_selection.
  intros p true_atoms Hin HinCand.
  unfold affine_all_paths_exhaustive_unsat_check_under_selection in Hcheck.
  apply forallb_forall with (x := p) in Hcheck.
  - eapply affine_path_exhaustive_unsat_check_under_selection_sound_candidates.
    + exact Hcheck.
    + exact HinCand.
  - exact Hin.
Qed.

(* ================================================================ *)
(* 13. Finite assignment representation for affine atoms             *)
(* ================================================================ *)

(* The exhaustive affine checker enumerates finite lists of true atoms.
   To connect this finite representation to an arbitrary Assignment rho,
   we need a small extensionality condition: rho must respect the atom
   equality used by affine_atom_eqb.

   This matters because affine_atom_eqb compares thresholds using Qeq_bool.
   Two rationals may be propositionally equal as Q values even if they are
   written differently. A completely arbitrary Assignment could distinguish
   them, so finite representation requires this compatibility assumption.
*)

Definition affine_assignment_respects_atom_eqb
           (rho : Assignment) : Prop :=
  forall a b : Atom,
    affine_atom_eqb a b = true ->
    rho a = rho b.

Fixpoint affine_true_atoms_of_assignment
         (rho : Assignment)
         (atoms : list Atom) : list Atom :=
  match atoms with
  | [] => []
  | a :: tl =>
      if rho a
      then a :: affine_true_atoms_of_assignment rho tl
      else affine_true_atoms_of_assignment rho tl
  end.

Lemma affine_atom_eqb_refl :
  forall a : Atom,
    affine_atom_eqb a a = true.
Proof.
  intros [f t].
  unfold affine_atom_eqb.
  simpl.
  rewrite Nat.eqb_refl.
  simpl.
  apply Qeq_bool_iff.
  reflexivity.
Qed.

Theorem affine_true_atoms_of_assignment_in_powerset :
  forall (rho : Assignment) (atoms : list Atom),
    In
      (affine_true_atoms_of_assignment rho atoms)
      (affine_powerset_atoms atoms).
Proof.
  intros rho atoms.
  induction atoms as [| a tl IH].
  - simpl.
    left.
    reflexivity.
  - simpl.
    destruct (rho a) eqn:Ha.
    + apply in_or_app.
      right.
      apply in_map.
      exact IH.
    + apply in_or_app.
      left.
      exact IH.
Qed.

Lemma affine_atom_in_bool_true_atoms_complete :
  forall (rho : Assignment) (atoms : list Atom) (a : Atom),
    In a atoms ->
    rho a = true ->
    affine_atom_in_bool
      a
      (affine_true_atoms_of_assignment rho atoms) = true.
Proof.
  intros rho atoms.
  induction atoms as [| b tl IH]; intros a Hin Hrho.
  - contradiction.
  - simpl in Hin.
    destruct Hin as [Heq | HinTl].
    + subst b.
      simpl.
      rewrite Hrho.
      simpl.
      rewrite affine_atom_eqb_refl.
      reflexivity.
    + simpl.
      destruct (rho b) eqn:Hb.
      * simpl.
        destruct (affine_atom_eqb a b) eqn:Hab.
        -- reflexivity.
        -- apply IH.
           ++ exact HinTl.
           ++ exact Hrho.
      * apply IH.
        -- exact HinTl.
        -- exact Hrho.
Qed.

Lemma affine_atom_in_bool_true_atoms_sound :
  forall (rho : Assignment) (atoms : list Atom) (a : Atom),
    affine_assignment_respects_atom_eqb rho ->
    affine_atom_in_bool
      a
      (affine_true_atoms_of_assignment rho atoms) = true ->
    rho a = true.
Proof.
  intros rho atoms.
  induction atoms as [| b tl IH]; intros a Hrespect HinBool.
  - simpl in HinBool.
    discriminate.
  - simpl in HinBool.
    simpl.
    destruct (rho b) eqn:Hb.
    + simpl in HinBool.
      destruct (affine_atom_eqb a b) eqn:Hab.
      * rewrite (Hrespect a b Hab).
        exact Hb.
      * apply IH.
        -- exact Hrespect.
        -- exact HinBool.
    + apply IH.
      * exact Hrespect.
      * exact HinBool.
Qed.

Theorem affine_true_atoms_assignment_agrees_on_atoms :
  forall (rho : Assignment) (atoms : list Atom) (a : Atom),
    affine_assignment_respects_atom_eqb rho ->
    In a atoms ->
    affine_assignment_from_true_atoms
      (affine_true_atoms_of_assignment rho atoms) a = rho a.
Proof.
  intros rho atoms a Hrespect Hin.
  unfold affine_assignment_from_true_atoms.
  destruct (rho a) eqn:Hrho.
  - apply affine_atom_in_bool_true_atoms_complete.
    + exact Hin.
    + exact Hrho.
  - destruct
      (affine_atom_in_bool
         a
         (affine_true_atoms_of_assignment rho atoms)) eqn:Hmem.
    + pose proof
        (affine_atom_in_bool_true_atoms_sound
           rho atoms a Hrespect Hmem)
        as Htrue.
      rewrite Hrho in Htrue.
      discriminate.
    + reflexivity.
Qed.

(* ================================================================ *)
(* 14. Assignment agreement preserves affine evaluation              *)
(* ================================================================ *)

Definition assignments_agree_on_atoms
           (atoms : list Atom)
           (rho sigma : Assignment) : Prop :=
  forall a : Atom,
    In a atoms ->
    rho a = sigma a.

Lemma signed_evalb_agree :
  forall (rho sigma : Assignment)
         (atoms : list Atom)
         (sa : SignedAtom),
    assignments_agree_on_atoms atoms rho sigma ->
    In sa.(sa_atom) atoms ->
    signed_evalb rho sa = signed_evalb sigma sa.
Proof.
  intros rho sigma atoms sa Hagree Hin.
  unfold signed_evalb.
  destruct sa as [a pos].
  simpl in *.
  rewrite (Hagree a Hin).
  reflexivity.
Qed.

Lemma affine_lhs_evalb_agree_terms :
  forall (rho sigma : Assignment)
         (terms : list SignedAtom),
    (forall sa : SignedAtom,
        In sa terms ->
        signed_evalb rho sa = signed_evalb sigma sa) ->
    affine_lhs_evalb rho terms = affine_lhs_evalb sigma terms.
Proof.
  intros rho sigma terms Hagree.
  unfold affine_lhs_evalb.
  induction terms as [| sa tl IH].
  - reflexivity.
  - simpl.
    rewrite Hagree.
    + rewrite IH.
      * reflexivity.
      * intros sa' Hin.
        apply Hagree.
        simpl.
        right.
        exact Hin.
    + simpl.
      left.
      reflexivity.
Qed.

Theorem affine_equation_evalb_agree :
  forall (rho sigma : Assignment)
         (e : AffineEquation),
    assignments_agree_on_atoms
      (affine_atoms_of_equation e) rho sigma ->
    affine_equation_evalb rho e =
    affine_equation_evalb sigma e.
Proof.
  intros rho sigma e Hagree.
  unfold affine_equation_evalb.
  f_equal.
  unfold affine_atoms_of_equation in Hagree.
  apply affine_lhs_evalb_agree_terms.
  intros sa Hin.
  apply signed_evalb_agree
    with (atoms := map sa_atom (ae_terms e)).
  - exact Hagree.
  - apply in_map.
    exact Hin.
Qed.

Lemma affine_system_evalb_agree :
  forall (rho sigma : Assignment)
         (eqs : list AffineEquation),
    assignments_agree_on_atoms
      (affine_atoms_of_system eqs) rho sigma ->
    affine_system_evalb rho eqs =
    affine_system_evalb sigma eqs.
Proof.
  intros rho sigma eqs Hagree.
  unfold affine_system_evalb.
  induction eqs as [| e tl IH].
  - reflexivity.
  - simpl.
    rewrite affine_equation_evalb_agree
      with (sigma := sigma).
    + rewrite IH.
      * reflexivity.
      * unfold assignments_agree_on_atoms.
        intros a Hin.
        apply Hagree.
        unfold affine_atoms_of_system.
        simpl.
        apply in_or_app.
        right.
        exact Hin.
    + unfold assignments_agree_on_atoms.
      intros a Hin.
      apply Hagree.
      unfold affine_atoms_of_system.
      simpl.
      apply in_or_app.
      left.
      exact Hin.
Qed.

Theorem affine_path_evalb_agree :
  forall (rho sigma : Assignment)
         (path : AffinePath),
    assignments_agree_on_atoms
      (affine_atoms_of_system (affine_path_equations path))
      rho sigma ->
    affine_path_evalb rho path =
    affine_path_evalb sigma path.
Proof.
  intros rho sigma path Hagree.
  rewrite <- affine_path_encoding_correct.
  rewrite <- affine_path_encoding_correct.
  apply affine_system_evalb_agree.
  exact Hagree.
Qed.
(* ================================================================ *)
(* 15. Completeness bridge from Assignment to finite candidates      *)
(* ================================================================ *)

(* Section 13 showed that any equality-compatible Assignment rho can be
   represented over a finite atom list by collecting exactly the atoms
   assigned true by rho.

   Section 14 showed that affine evaluation is preserved when assignments
   agree on the atoms used by the system.

   This section combines both results: if an arbitrary compatible rho
   satisfies an affine system/path, then the exhaustive finite checker has
   a corresponding finite candidate.
*)

Theorem affine_system_evalb_true_atoms_of_assignment :
  forall (rho : Assignment) (eqs : list AffineEquation),
    affine_assignment_respects_atom_eqb rho ->
    affine_system_evalb
      (affine_assignment_from_true_atoms
         (affine_true_atoms_of_assignment
            rho
            (affine_atoms_of_system eqs)))
      eqs
    =
    affine_system_evalb rho eqs.
Proof.
  intros rho eqs Hrespect.
  apply affine_system_evalb_agree.
  unfold assignments_agree_on_atoms.
  intros a Hin.
  apply affine_true_atoms_assignment_agrees_on_atoms.
  - exact Hrespect.
  - exact Hin.
Qed.

Theorem find_affine_assignment_under_selection_from_candidates_complete_candidate :
  forall (x0 : Valuation)
         (S : list Feature)
         (atoms : list Atom)
         (eqs : list AffineEquation)
         (candidates : list (list Atom))
         (witness : list Atom),
    In witness candidates ->
    affine_system_evalb_under_selection
      x0 S atoms eqs witness = true ->
    exists found : list Atom,
      find_affine_assignment_under_selection_from_candidates
        x0 S atoms eqs candidates = Some found.
Proof.
  intros x0 S atoms eqs candidates.
  induction candidates as [| cand tl IH]; intros witness Hin Hwit.
  - contradiction.
  - simpl in Hin.
    simpl.
    destruct
      (affine_system_evalb_under_selection
         x0 S atoms eqs cand) eqn:Hcand.
    + exists cand.
      reflexivity.
    + destruct Hin as [Heq | HinTl].
      * subst witness.
        rewrite Hwit in Hcand.
        discriminate.
      * apply (IH witness).
        -- exact HinTl.
        -- exact Hwit.
Qed.

Theorem find_affine_satisfying_assignment_under_selection_complete_from_assignment :
  forall (x0 : Valuation)
         (S : list Feature)
         (eqs : list AffineEquation)
         (rho : Assignment),
    affine_assignment_respects_atom_eqb rho ->
    affine_selected_candidate_agreementb
      x0 S
      (affine_atoms_of_system eqs)
      (affine_true_atoms_of_assignment
         rho
         (affine_atoms_of_system eqs)) = true ->
    affine_system_evalb rho eqs = true ->
    exists found : list Atom,
      find_affine_satisfying_assignment_under_selection
        x0 S eqs (affine_atoms_of_system eqs) = Some found.
Proof.
  intros x0 S eqs rho Hrespect Hsel Hsys.
  unfold find_affine_satisfying_assignment_under_selection.
  apply find_affine_assignment_under_selection_from_candidates_complete_candidate
    with
      (witness :=
         affine_true_atoms_of_assignment
           rho
           (affine_atoms_of_system eqs)).
  - apply affine_true_atoms_of_assignment_in_powerset.
  - unfold affine_system_evalb_under_selection.
    apply andb_true_iff.
    split.
    + exact Hsel.
    + rewrite affine_system_evalb_true_atoms_of_assignment.
      * exact Hsys.
      * exact Hrespect.
Qed.

Theorem find_affine_path_assignment_under_selection_complete_from_assignment :
  forall (x0 : Valuation)
         (S : list Feature)
         (path : AffinePath)
         (rho : Assignment),
    affine_assignment_respects_atom_eqb rho ->
    affine_selected_candidate_agreementb
      x0 S
      (affine_atoms_of_system (affine_path_equations path))
      (affine_true_atoms_of_assignment
         rho
         (affine_atoms_of_system (affine_path_equations path))) = true ->
    affine_path_evalb rho path = true ->
    exists found : list Atom,
      find_affine_satisfying_assignment_under_selection
        x0 S
        (affine_path_equations path)
        (affine_atoms_of_system (affine_path_equations path))
      = Some found.
Proof.
  intros x0 S path rho Hrespect Hsel Hpath.
  assert
    (Hsys :
       affine_system_evalb rho (affine_path_equations path) = true).
  {
    rewrite affine_path_encoding_correct.
    exact Hpath.
  }
  exact
    (find_affine_satisfying_assignment_under_selection_complete_from_assignment
       x0
       S
       (affine_path_equations path)
       rho
       Hrespect
       Hsel
       Hsys).
Qed.

Theorem affine_path_exhaustive_unsat_check_under_selection_no_respecting_assignment :
  forall (x0 : Valuation)
         (S : list Feature)
         (path : AffinePath)
         (rho : Assignment),
    affine_path_exhaustive_unsat_check_under_selection
      x0 S path = true ->
    affine_assignment_respects_atom_eqb rho ->
    affine_selected_candidate_agreementb
      x0 S
      (affine_atoms_of_system (affine_path_equations path))
      (affine_true_atoms_of_assignment
         rho
         (affine_atoms_of_system (affine_path_equations path))) = true ->
    affine_path_evalb rho path = false.

Proof.
  intros x0 S path rho Hcheck Hrespect Hsel.

  set
    (cand :=
       affine_true_atoms_of_assignment
         rho
         (affine_atoms_of_system (affine_path_equations path))).

  assert
    (HinCand :
       In cand
          (affine_powerset_atoms
             (affine_atoms_of_system (affine_path_equations path)))).
  {
    unfold cand.
    apply affine_true_atoms_of_assignment_in_powerset.
  }

  assert
    (HselCand :
       affine_selected_candidate_agreementb
         x0 S
         (affine_atoms_of_system (affine_path_equations path))
         cand = true).
  {
    unfold cand.
    exact Hsel.
  }

  pose proof
    (affine_path_exhaustive_unsat_check_under_selection_sound_candidates
       x0 S path cand Hcheck HinCand)
    as Hblocked.

  unfold affine_selected_candidate_path_evalb in Hblocked.
  rewrite HselCand in Hblocked.
  simpl in Hblocked.

  assert
    (Hagree :
       assignments_agree_on_atoms
         (affine_atoms_of_system (affine_path_equations path))
         (affine_assignment_from_true_atoms cand)
         rho).
  {
    unfold assignments_agree_on_atoms.
    intros a Hin.
    unfold cand.
    apply affine_true_atoms_assignment_agrees_on_atoms.
    - exact Hrespect.
    - exact Hin.
  }

  pose proof
    (affine_path_evalb_agree
       (affine_assignment_from_true_atoms cand)
       rho
       path
       Hagree)
    as HevalAgree.

  rewrite HevalAgree in Hblocked.
  exact Hblocked.
Qed.


(* ================================================================ *)
(* 16. Prop-level selected agreement implies Boolean agreement       *)
(* ================================================================ *)

Lemma affine_feature_in_bool_true_in :
  forall (f : Feature) (S : list Feature),
    affine_feature_in_bool f S = true ->
    In f S.
Proof.
  intros f S.
  induction S as [| g tl IH]; intros H.
  - simpl in H.
    discriminate.
  - simpl in H.
    destruct (Nat.eqb f g) eqn:Heq.
    + left.
      apply Nat.eqb_eq in Heq.
      subst f.
      reflexivity.
    + right.
      apply IH.
      exact H.
Qed.

Theorem affine_selected_candidate_agreementb_from_prop :
  forall (x0 : Valuation)
         (S : list Feature)
         (atoms : list Atom)
         (rho : Assignment),
    affine_assignment_respects_atom_eqb rho ->
    affine_selected_atom_agreement x0 S atoms rho ->
    affine_selected_candidate_agreementb
      x0 S atoms
      (affine_true_atoms_of_assignment rho atoms) = true.
Proof.
  intros x0 S atoms rho Hrespect Hagree.
  unfold affine_selected_candidate_agreementb.
  apply forallb_forall.
  intros [f t] Hin.
  destruct (affine_feature_in_bool f S) eqn:Hfeat.
  - apply Bool.eqb_true_iff.
    pose proof
      (affine_true_atoms_assignment_agrees_on_atoms
         rho atoms (f,t) Hrespect Hin)
      as Hcand.
    rewrite Hcand.
    symmetry.
    apply Hagree.
    + exact Hin.
    + apply affine_feature_in_bool_true_in.
      exact Hfeat.
  - reflexivity.
Qed.

Theorem affine_path_exhaustive_unsat_check_under_selection_sound_compatible_assignment :
  forall (x0 : Valuation)
         (S : list Feature)
         (path : AffinePath)
         (rho : Assignment),
    affine_path_exhaustive_unsat_check_under_selection
      x0 S path = true ->
    affine_assignment_respects_atom_eqb rho ->
    affine_selected_atom_agreement
      x0 S
      (affine_atoms_of_system (affine_path_equations path))
      rho ->
    affine_path_evalb rho path = false.
Proof.
  intros x0 S path rho Hcheck Hrespect Hagree.
  apply
    (affine_path_exhaustive_unsat_check_under_selection_no_respecting_assignment
       x0 S path rho Hcheck Hrespect).
  apply affine_selected_candidate_agreementb_from_prop.
  - exact Hrespect.
  - exact Hagree.
Qed.

Definition affine_path_satisfiable_under_selection_compatible
           (x0 : Valuation)
           (S : list Feature)
           (path : AffinePath) : Prop :=
  exists rho : Assignment,
    affine_assignment_respects_atom_eqb rho
    /\
    affine_selected_atom_agreement
      x0 S
      (affine_atoms_of_system (affine_path_equations path))
      rho
    /\
    affine_path_evalb rho path = true.

Theorem affine_path_exhaustive_unsat_check_under_selection_blocks_compatible_satisfiability :
  forall (x0 : Valuation)
         (S : list Feature)
         (path : AffinePath),
    affine_path_exhaustive_unsat_check_under_selection
      x0 S path = true ->
    ~ affine_path_satisfiable_under_selection_compatible x0 S path.
Proof.
  intros x0 S path Hcheck Hsat.
  unfold affine_path_satisfiable_under_selection_compatible in Hsat.
  destruct Hsat as [rho [Hrespect [Hagree Hpath]]].
  pose proof
    (affine_path_exhaustive_unsat_check_under_selection_sound_compatible_assignment
       x0 S path rho Hcheck Hrespect Hagree)
    as Hfalse.
  rewrite Hpath in Hfalse.
  discriminate.
Qed.

Definition affine_compatible_opposite_paths_blocked_under_selection
           (x0 : Valuation)
           (S : list Feature)
           (opposite_paths : list AffinePath) : Prop :=
  forall p : AffinePath,
    In p opposite_paths ->
    ~ affine_path_satisfiable_under_selection_compatible x0 S p.

Theorem affine_all_paths_exhaustive_unsat_check_under_selection_blocks_compatible_paths :
  forall (x0 : Valuation)
         (S : list Feature)
         (opposite_paths : list AffinePath),
    affine_all_paths_exhaustive_unsat_check_under_selection
      x0 S opposite_paths = true ->
    affine_compatible_opposite_paths_blocked_under_selection
      x0 S opposite_paths.
Proof.
  intros x0 S opposite_paths Hcheck.
  unfold affine_compatible_opposite_paths_blocked_under_selection.
  intros p Hin Hsat.
  unfold affine_all_paths_exhaustive_unsat_check_under_selection in Hcheck.
  apply forallb_forall with (x := p) in Hcheck.
  - apply
      (affine_path_exhaustive_unsat_check_under_selection_blocks_compatible_satisfiability
         x0 S p Hcheck).
    exact Hsat.
  - exact Hin.
Qed.

(* ================================================================ *)
(* 17. GF(2) normalization of affine equations                       *)
(* ================================================================ *)

(* A signed affine equation may contain positive atoms B and negative
   atoms not B. A GF(2) linear solver should operate on raw positive
   Boolean atoms only.

   Boolean identity:
       negb x = x xor true

   Therefore each negative signed atom flips the right-hand side.
*)

Record GF2Equation := {
  gf2_terms : list Atom;
  gf2_rhs   : bool
}.

Definition gf2_lhs_evalb
           (rho : Assignment)
           (terms : list Atom) : bool :=
  xor_list (map rho terms).

Definition gf2_equation_evalb
           (rho : Assignment)
           (e : GF2Equation) : bool :=
  Bool.eqb (gf2_lhs_evalb rho e.(gf2_terms)) e.(gf2_rhs).

Fixpoint gf2_normalize_terms
         (terms : list SignedAtom) : list Atom * bool :=
  match terms with
  | [] => ([], false)
  | sa :: tl =>
      let '(vars, flip) := gf2_normalize_terms tl in
      (sa.(sa_atom) :: vars,
       if sa.(sa_positive) then flip else negb flip)
  end.

Definition gf2_of_affine_equation
           (e : AffineEquation) : GF2Equation :=
  let '(vars, flip) := gf2_normalize_terms e.(ae_terms) in
  {|
    gf2_terms := vars;
    gf2_rhs   := xorb e.(ae_rhs) flip
  |}.

Lemma gf2_normalize_terms_correct_pair :
  forall (rho : Assignment)
         (terms : list SignedAtom)
         (vars : list Atom)
         (flip : bool),
    gf2_normalize_terms terms = (vars, flip) ->
    affine_lhs_evalb rho terms =
    xorb (gf2_lhs_evalb rho vars) flip.
Proof.
  intros rho terms.
  induction terms as [| sa tl IH]; intros vars flip Hnf.
  - simpl in Hnf.
    inversion Hnf.
    subst.
    reflexivity.
  - destruct sa as [a pos].
    simpl in Hnf.
    destruct (gf2_normalize_terms tl) as [vars_t flip_t] eqn:Ht.
    inversion Hnf.
    subst vars flip.
    clear Hnf.

    specialize (IH vars_t flip_t eq_refl).

    unfold affine_lhs_evalb in *.
    unfold gf2_lhs_evalb in *.
    simpl in *.
    unfold signed_evalb in *.
    simpl in *.

    rewrite IH.

    destruct pos;
      destruct (rho a);
      destruct (xor_list (map rho vars_t));
      destruct flip_t;
      reflexivity.
Qed.

Lemma gf2_normalize_terms_correct :
  forall (rho : Assignment) (terms : list SignedAtom),
    let nf := gf2_normalize_terms terms in
    affine_lhs_evalb rho terms =
    xorb (gf2_lhs_evalb rho (fst nf)) (snd nf).
Proof.
  intros rho terms.
  destruct (gf2_normalize_terms terms) as [vars flip] eqn:Hnf.
  simpl.
  eapply gf2_normalize_terms_correct_pair.
  exact Hnf.
Qed.

Lemma gf2_of_affine_equation_correct_pair :
  forall (rho : Assignment)
         (terms : list SignedAtom)
         (rhs : bool)
         (vars : list Atom)
         (flip : bool),
    gf2_normalize_terms terms = (vars, flip) ->
    gf2_equation_evalb
      rho
      {| gf2_terms := vars;
         gf2_rhs := xorb rhs flip |}
    =
    affine_equation_evalb
      rho
      {| ae_terms := terms;
         ae_rhs := rhs |}.
Proof.
  intros rho terms rhs vars flip Hnf.
  unfold gf2_equation_evalb.
  unfold affine_equation_evalb.
  simpl.

  assert
    (Hnorm :
       affine_lhs_evalb rho terms =
       xorb (gf2_lhs_evalb rho vars) flip).
  {
    eapply gf2_normalize_terms_correct_pair.
    exact Hnf.
  }

  rewrite Hnorm.

  destruct (gf2_lhs_evalb rho vars);
    destruct rhs;
    destruct flip;
    reflexivity.
Qed.

Theorem gf2_of_affine_equation_correct :
  forall (rho : Assignment) (e : AffineEquation),
    gf2_equation_evalb rho (gf2_of_affine_equation e)
    =
    affine_equation_evalb rho e.
Proof.
  intros rho [terms rhs].
  unfold gf2_of_affine_equation.
  cbn.
  destruct (gf2_normalize_terms terms) as [vars flip] eqn:Hnf.
  cbn.
  exact
    (gf2_of_affine_equation_correct_pair
       rho terms rhs vars flip Hnf).
Qed.

Definition gf2_system_evalb
           (rho : Assignment)
           (eqs : list GF2Equation) : bool :=
  forallb (gf2_equation_evalb rho) eqs.

Definition gf2_of_affine_system
           (eqs : list AffineEquation) : list GF2Equation :=
  map gf2_of_affine_equation eqs.

Theorem gf2_of_affine_system_correct :
  forall (rho : Assignment) (eqs : list AffineEquation),
    gf2_system_evalb rho (gf2_of_affine_system eqs)
    =
    affine_system_evalb rho eqs.
Proof.
  intros rho eqs.
  induction eqs as [| e tl IH].
  - reflexivity.
  - unfold gf2_of_affine_system.
    simpl.
    unfold gf2_system_evalb.
    simpl.
    rewrite gf2_of_affine_equation_correct.
    unfold affine_system_evalb.
    simpl.
    fold (affine_system_evalb rho tl).
    fold (gf2_of_affine_system tl).
    fold (gf2_system_evalb rho (gf2_of_affine_system tl)).
    rewrite IH.
    reflexivity.
Qed.

Definition gf2_of_affine_path
           (path : AffinePath) : list GF2Equation :=
  gf2_of_affine_system (affine_path_equations path).

Theorem gf2_of_affine_path_correct :
  forall (rho : Assignment) (path : AffinePath),
    gf2_system_evalb rho (gf2_of_affine_path path)
    =
    affine_path_evalb rho path.
Proof.
  intros rho path.
  unfold gf2_of_affine_path.
  rewrite gf2_of_affine_system_correct.
  apply affine_path_encoding_correct.
Qed.


(* ================================================================ *)
(* 18. GF(2) row-operation preservation                             *)
(* ================================================================ *)

(* Gaussian elimination over GF(2) uses elementary row operations.
   The key operation is:

       row_i := row_i xor row_j

   This section proves that this operation preserves the solution set,
   provided the pivot/source row remains in the system.
*)

Lemma xor_list_app :
  forall l1 l2 : list bool,
    xor_list (l1 ++ l2) =
    xorb (xor_list l1) (xor_list l2).
Proof.
  intros l1 l2.
  induction l1 as [| b tl IH].
  - reflexivity.
  - simpl.
    rewrite IH.
    destruct b;
      destruct (xor_list tl);
      destruct (xor_list l2);
      reflexivity.
Qed.

Lemma gf2_lhs_evalb_app :
  forall (rho : Assignment) (l1 l2 : list Atom),
    gf2_lhs_evalb rho (l1 ++ l2) =
    xorb (gf2_lhs_evalb rho l1)
         (gf2_lhs_evalb rho l2).
Proof.
  intros rho l1 l2.
  unfold gf2_lhs_evalb.
  rewrite map_app.
  apply xor_list_app.
Qed.

Definition gf2_xor_equation
           (e1 e2 : GF2Equation) : GF2Equation :=
  {|
    gf2_terms := e1.(gf2_terms) ++ e2.(gf2_terms);
    gf2_rhs   := xorb e1.(gf2_rhs) e2.(gf2_rhs)
  |}.

Theorem gf2_row_xor_preserves_with_pivot :
  forall (rho : Assignment)
         (e1 e2 : GF2Equation),
    gf2_equation_evalb rho e2 = true ->
    gf2_equation_evalb rho (gf2_xor_equation e1 e2)
    =
    gf2_equation_evalb rho e1.
Proof.
  intros rho [terms1 rhs1] [terms2 rhs2] He2.
  unfold gf2_xor_equation in *.
  unfold gf2_equation_evalb in *.
  simpl in *.
  rewrite gf2_lhs_evalb_app.

  destruct (gf2_lhs_evalb rho terms1);
    destruct (gf2_lhs_evalb rho terms2);
    destruct rhs1;
    destruct rhs2;
    simpl in *;
    try discriminate;
    reflexivity.
Qed.

Theorem gf2_row_xor_sound :
  forall (rho : Assignment)
         (e1 e2 : GF2Equation),
    gf2_equation_evalb rho e1 = true ->
    gf2_equation_evalb rho e2 = true ->
    gf2_equation_evalb rho (gf2_xor_equation e1 e2) = true.
Proof.
  intros rho e1 e2 H1 H2.
  rewrite gf2_row_xor_preserves_with_pivot.
  - exact H1.
  - exact H2.
Qed.

Theorem gf2_row_xor_recover_left :
  forall (rho : Assignment)
         (e1 e2 : GF2Equation),
    gf2_equation_evalb rho e2 = true ->
    gf2_equation_evalb rho (gf2_xor_equation e1 e2) = true ->
    gf2_equation_evalb rho e1 = true.
Proof.
  intros rho e1 e2 H2 Hxor.
  rewrite <- (gf2_row_xor_preserves_with_pivot rho e1 e2 H2).
  exact Hxor.
Qed.

Theorem gf2_system_row_xor_first_preserves :
  forall (rho : Assignment)
         (e1 e2 : GF2Equation)
         (rest : list GF2Equation),
    gf2_system_evalb rho (e1 :: e2 :: rest)
    =
    gf2_system_evalb rho ((gf2_xor_equation e1 e2) :: e2 :: rest).
Proof.
  intros rho e1 e2 rest.
  unfold gf2_system_evalb.
  simpl.
  destruct (gf2_equation_evalb rho e2) eqn:He2.
  - rewrite (gf2_row_xor_preserves_with_pivot rho e1 e2 He2).
    reflexivity.
  - destruct (gf2_equation_evalb rho e1);
      destruct (gf2_equation_evalb rho (gf2_xor_equation e1 e2));
      reflexivity.
Qed.

Theorem gf2_system_swap_first_two_preserves :
  forall (rho : Assignment)
         (e1 e2 : GF2Equation)
         (rest : list GF2Equation),
    gf2_system_evalb rho (e1 :: e2 :: rest)
    =
    gf2_system_evalb rho (e2 :: e1 :: rest).
Proof.
  intros rho e1 e2 rest.
  unfold gf2_system_evalb.
  simpl.
  destruct (gf2_equation_evalb rho e1);
    destruct (gf2_equation_evalb rho e2);
    reflexivity.
Qed.

Theorem gf2_system_duplicate_first_preserves :
  forall (rho : Assignment)
         (e : GF2Equation)
         (rest : list GF2Equation),
    gf2_system_evalb rho (e :: e :: rest)
    =
    gf2_system_evalb rho (e :: rest).
Proof.
  intros rho e rest.
  unfold gf2_system_evalb.
  simpl.
  destruct (gf2_equation_evalb rho e);
    reflexivity.
Qed.
(* ================================================================ *)
(* 19. GF(2) duplicate cancellation and empty-row facts              *)
(* ================================================================ *)

(* In GF(2), duplicated variables cancel:
       x xor x = 0

   This section proves the semantic facts needed before implementing
   row normalization / Gaussian elimination.
*)

Lemma xorb_same_cancel_left :
  forall b c : bool,
    xorb b (xorb b c) = c.
Proof.
  intros b c.
  destruct b, c; reflexivity.
Qed.

Lemma gf2_lhs_evalb_cancel_identical_adjacent :
  forall (rho : Assignment) (a : Atom) (rest : list Atom),
    gf2_lhs_evalb rho (a :: a :: rest)
    =
    gf2_lhs_evalb rho rest.
Proof.
  intros rho a rest.
  unfold gf2_lhs_evalb.
  simpl.
  rewrite xorb_same_cancel_left.
  reflexivity.
Qed.

Theorem gf2_equation_cancel_identical_adjacent :
  forall (rho : Assignment)
         (a : Atom)
         (rest : list Atom)
         (rhs : bool),
    gf2_equation_evalb
      rho
      {| gf2_terms := a :: a :: rest;
         gf2_rhs := rhs |}
    =
    gf2_equation_evalb
      rho
      {| gf2_terms := rest;
         gf2_rhs := rhs |}.
Proof.
  intros rho a rest rhs.
  unfold gf2_equation_evalb.
  simpl.
  rewrite gf2_lhs_evalb_cancel_identical_adjacent.
  reflexivity.
Qed.

Theorem gf2_equation_cancel_eqb_adjacent :
  forall (rho : Assignment)
         (a b : Atom)
         (rest : list Atom)
         (rhs : bool),
    affine_assignment_respects_atom_eqb rho ->
    affine_atom_eqb a b = true ->
    gf2_equation_evalb
      rho
      {| gf2_terms := a :: b :: rest;
         gf2_rhs := rhs |}
    =
    gf2_equation_evalb
      rho
      {| gf2_terms := rest;
         gf2_rhs := rhs |}.
Proof.
  intros rho a b rest rhs Hrespect Hab.
  unfold gf2_equation_evalb.
  unfold gf2_lhs_evalb.
  simpl.
  rewrite (Hrespect a b Hab).
  rewrite xorb_same_cancel_left.
  reflexivity.
Qed.

Lemma gf2_lhs_evalb_empty :
  forall rho : Assignment,
    gf2_lhs_evalb rho [] = false.
Proof.
  intros rho.
  reflexivity.
Qed.

Theorem gf2_empty_false_equation_true :
  forall rho : Assignment,
    gf2_equation_evalb
      rho
      {| gf2_terms := [];
         gf2_rhs := false |}
    = true.
Proof.
  intros rho.
  reflexivity.
Qed.

Theorem gf2_empty_true_equation_false :
  forall rho : Assignment,
    gf2_equation_evalb
      rho
      {| gf2_terms := [];
         gf2_rhs := true |}
    = false.
Proof.
  intros rho.
  reflexivity.
Qed.

Definition gf2_empty_contradiction
           (e : GF2Equation) : bool :=
  match e.(gf2_terms), e.(gf2_rhs) with
  | [], true => true
  | _, _ => false
  end.

Theorem gf2_empty_contradiction_unsat :
  forall (rho : Assignment) (e : GF2Equation),
    gf2_empty_contradiction e = true ->
    gf2_equation_evalb rho e = false.
Proof.
  intros rho [terms rhs] Hcontra.
  unfold gf2_empty_contradiction in Hcontra.
  destruct terms as [| a tl].
  - destruct rhs.
    + simpl.
      reflexivity.
    + simpl in Hcontra.
      discriminate.
  - simpl in Hcontra.
    discriminate.
Qed.