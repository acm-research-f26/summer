using UnityEngine;

public class DiamondScript : MonoBehaviour
{
    public Sprite destroyedSprite;
    SpriteRenderer renderer;
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        renderer = GetComponent<SpriteRenderer>();
        renderer.enabled = false;
        LeverScript.leverPressed += OnLeverPress;
    }

    void OnTriggerEnter2D(Collider2D collision)
    {
        if(collision.gameObject.name == "Player" && GameManagerScript.instance.hasHammer && GameManagerScript.instance.leverPressed)
        {
            renderer.sprite = destroyedSprite;
            GameManagerScript.instance.diamondStolen = true;
        }
    }

    void OnLeverPress()
    {
        renderer.enabled = true;
    }
}
