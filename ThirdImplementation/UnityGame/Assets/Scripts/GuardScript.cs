using UnityEngine;

public enum CurrentAction
{
    MoveToPlayerOrLastPointSpotted = 1,
    RaiseAlarm = 2,
    InvestigateSound = 3,
    WanderToRandomPlace = 4
};

public class GuardScript : MonoBehaviour
{
    CurrentAction currentBehavior;
    Vector2 lastPlayerPointSpotted;
    Vector2 targetPosition;
    Vector2 lastSound;
    float movementSpeed;
    public AudioClip angryMusic;
    Rigidbody2D rb;
    AudioSource audiosource;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        currentBehavior = CurrentAction.WanderToRandomPlace;
        lastPlayerPointSpotted = transform.position;
        targetPosition = new Vector2(Random.Range(-73f, 180f), transform.position.y);
        lastSound = transform.position;
        audiosource = GetComponent<AudioSource>();

        movementSpeed = 5;

        rb = GetComponent<Rigidbody2D>();

        GameManagerScript.lockdownInitiated += OnLockdown;
    }

    CurrentAction PickNewAction()
    {
        CurrentAction chosenAction = (CurrentAction) Random.Range(1, 4);
        if(chosenAction == CurrentAction.WanderToRandomPlace)
        {
            targetPosition = new Vector2(Random.Range(-73f, 180f), transform.position.y);
        }

        return chosenAction;

    }

    void DoMoveAction()
    {
        Vector2 movingPosition = Vector2.MoveTowards(transform.position, targetPosition, movementSpeed * Time.deltaTime);
        rb.MovePosition(movingPosition);
        if(Vector2.Distance(movingPosition, targetPosition) < 15f)
        {
            currentBehavior = PickNewAction();
        }
    }

    // Update is called once per frame
    void Update()
    {
        switch (currentBehavior)
        {
            case CurrentAction.MoveToPlayerOrLastPointSpotted:
                targetPosition = lastPlayerPointSpotted;
                DoMoveAction();
                break;
            case CurrentAction.RaiseAlarm:
                GameManagerScript.instance.RaiseAlarm();
                currentBehavior = PickNewAction();
                break;
            case CurrentAction.InvestigateSound:
                targetPosition = lastSound;
                DoMoveAction();
                break;
            case CurrentAction.WanderToRandomPlace:
                DoMoveAction();
                break;
            default:
                break;
        }
        if(rb.linearVelocityX < 0)
        {
            transform.localScale = new Vector2(-1, transform.localScale.y);
        }
        else if(rb.linearVelocityX > 0)
        {
            transform.localScale = new Vector2(1, transform.localScale.y);
        }
    }

    void OnLockdown()
    {
        movementSpeed *= 2;
        audiosource.Stop();
        audiosource.clip = angryMusic;
        audiosource.volume = 0.5f;
        audiosource.Play();
    }

    void OnTriggerEnter2D(Collider2D collision)
    {
        if(collision.gameObject.name == "Player" && GameManagerScript.instance.inLockdown)
        {
            GameManagerScript.instance.LoseGame();
        }
    }
}
