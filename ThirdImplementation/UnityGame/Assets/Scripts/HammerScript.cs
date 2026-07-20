using UnityEngine;

public class HammerScript : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        float newXPos = Random.Range(-70, 70);

        transform.position = new Vector2(newXPos, transform.position.y);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnTriggerEnter2D(Collider2D collision)
    {
        Debug.Log($"object name is {collision.gameObject.name}");
        if(collision.gameObject.name == "Player")
        {
            Destroy(gameObject);
            GameManagerScript.instance.hasHammer = true;
        }
    }
}
