using UnityEngine;

public class VaseScript : MonoBehaviour
{
    public Sprite destroyedSprite;

    SpriteRenderer renderer;
    AudioSource soundController;
    bool alreadyDestroyed;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        renderer = GetComponent<SpriteRenderer>();
        soundController = GetComponent<AudioSource>();
        alreadyDestroyed = false;
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
