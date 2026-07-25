using UnityEngine;

public class VaseScript : MonoBehaviour
{
    public GuardScript guard;
    public Sprite destroyedSprite;

    SpriteRenderer renderer;
    AudioSource soundController;
    public bool alreadyDestroyed;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    
    public bool timeToFindCulpritPassed;

    public float timeSinceDestroyed;

    public const float timeToFindCulprit = 5;
    void Start()
    {
        renderer = GetComponent<SpriteRenderer>();
        soundController = GetComponent<AudioSource>();
        alreadyDestroyed = false;

        timeSinceDestroyed = 0;

        timeToFindCulpritPassed = false;
    }

    void Update()
    {
        if(alreadyDestroyed && timeSinceDestroyed < timeToFindCulprit)
        {
            timeSinceDestroyed += Time.deltaTime;
            if(timeSinceDestroyed >= timeToFindCulprit)
            {
                timeToFindCulpritPassed = true;
                guard.RemoveVaseFromMemory(this);
            }
        }
    }

    void OnTriggerEnter2D(Collider2D collision)
    {
        if(collision.gameObject.name == "Player" && !alreadyDestroyed)
        {
            alreadyDestroyed = true;
            renderer.sprite = destroyedSprite;
            soundController.Play();
            GameManagerScript.soundOccurred.Invoke(transform.position);
            transform.position = new Vector2(transform.position.x + 1, transform.position.y);
        }
    }
}
