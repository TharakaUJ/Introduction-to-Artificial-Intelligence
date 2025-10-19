(define (domain move-domain)
  (:requirements :strips :typing)
  (:types room)
  
  (:predicates
    (at ?r - room)           ; the agent is in room ?r
    (connected ?r1 ?r2 - room) ; room ?r1 is connected to room ?r2
  )

  (:action move
    :parameters (?from ?to - room)
    :precondition (and (at ?from) (connected ?from ?to))
    :effect (and
              (not (at ?from))
              (at ?to)
            )
  )
)

