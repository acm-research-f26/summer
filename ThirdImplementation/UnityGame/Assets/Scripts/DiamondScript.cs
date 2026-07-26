using UnityEngine;

public class DiamondScript : MonoBehaviour
{
    public Sprite destroyedSprite;
    SpriteRenderer renderer;

    AudioSource audioController;
    public bool isBroken;
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        renderer = GetComponent<SpriteRenderer>();
        audioController = GetComponent<AudioSource>();
        renderer.enabled = false;
        LeverScript.leverPressed += OnLeverPress;
        isBroken = false;
    }

    void OnTriggerEnter2D(Collider2D collision)
    {
        if(collision.gameObject.name == "Player" && GameManagerScript.instance.hasHammer && GameManagerScript.instance.leverPressed && !GameManagerScript.instance.diamondStolen)
        {
            renderer.sprite = destroyedSprite;
            isBroken = true;
            GameManagerScript.instance.diamondStolen = true;
            audioController.Play();
            GameManagerScript.soundOccurred.Invoke(transform.position);
        }
    }

    void OnLeverPress()
    {
        renderer.enabled = true;

    }
}
