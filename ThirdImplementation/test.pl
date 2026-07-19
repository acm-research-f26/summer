:- use_module(library(scasp)).
:- use_module(library(http/json)).

danger(enemy) :-
    nearby(enemy),
    armed(enemy).

nearby(enemy).
armed(enemy).

should_raise_alarm() :- 

%! term_to_json(+Term, -Json) is det.
%  Recursively converts a Prolog term into a JSON-safe structure.
term_to_json(Term, Json) :-
    ( is_list(Term)
    -> maplist(term_to_json, Term, Json)
    ;  compound(Term)
    -> Term =.. [F|Args],
       maplist(term_to_json, Args, ArgsJson),
       Json = _{functor: F, args: ArgsJson}
    ;  Json = Term
    ).

bindings_to_json([], _{}).
bindings_to_json([Name=Value|T], Json) :-
    term_to_json(Value, ValueJson),
    bindings_to_json(T, RestJson),
    Json = RestJson.put(Name, ValueJson).

main :-
    current_prolog_flag(argv, [QueryAtom]),
    catch(
        (   term_string(Query, QueryAtom, [variable_names(Bindings)]),
            (   scasp(Query, [model(Model)])
            ->  bindings_to_json(Bindings, BindingsJson),
                term_to_json(Model, ModelJson),
                Result = _{ok: true, bindings: BindingsJson, model: ModelJson}
            ;   Result = _{ok: false, error: "no solution"}
            )
        ),
        Error,
        ( term_string(Error, ErrStr),
          Result = _{ok: false, error: ErrStr}
        )
    ),
    json_write_dict(current_output, Result),
    nl.

:- initialization(main, main).