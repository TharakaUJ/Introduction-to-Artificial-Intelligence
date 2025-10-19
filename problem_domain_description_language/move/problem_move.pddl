(define (problem move-problem)
  (:domain move-domain)

  (:objects
    A B - room
  )

  (:init
    (connected A B)
    (connected B A)
    (at A)
  )

  (:goal
    (at B)
  )
)

